"""Own-growth visibility, policy likelihoods, recovery, and evaluation plumbing."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import taverngym as tg

from training.attachments import policy_kwargs
from training.common import load_checkpoint, tensors
from training.entity import EntityPolicy
from training.growth import FEATURE_NAMES, N_FEATURES, from_views
from training.league import LeagueEnv, observation_rows
from training.strategy import ExpertStrategyPolicy, StrategyTeacher, add_public_context, add_strategy_observation
from test_attachments import fixture
from test_strategy import book


def pair():
    game = fixture()
    game.debug(json.dumps(dict(cmd="add_minion", p=0, zone="hand", card="BG25_009")))
    state = json.loads(game.state_json())
    state["players"][0]["stats"]["eternal_knight_deaths"] = 40
    return game, tg.Game.from_json(json.dumps(state))


def observations(games, policy):
    rows = [(i, 0) for i in range(len(games))]
    obs = observation_rows(games, rows, public_attachments=policy.requires_attachments)
    add_strategy_observation(policy, obs, SimpleNamespace(games=games), rows)
    return obs


def test_feature_order_is_a_stable_checkpoint_contract_and_only_own_allowlisted_data_enters():
    assert FEATURE_NAMES == (
        "eternal_knight_deaths", "knight_bonus", "undead_atk", "undead_hp",
        "beetle_atk", "beetle_hp", "volumizer_atk", "volumizer_hp",
        "elem_give_atk", "elem_give_hp", "tavern_elemental_atk", "tavern_elemental_hp",
        "baller_bonus", "lobster_bonus", "magnetized", "spellcraft_cast")
    view = json.loads(pair()[1].strategy_view(0))
    expected = from_views([view])
    changed = deepcopy(view)
    changed["opponents"] = [{"stats": {k: 999999 for k in FEATURE_NAMES}, "hand": ["private"]}]
    changed["me"]["stats"]["unlisted_private_field"] = 999999
    np.testing.assert_array_equal(expected, from_views([changed]))
    assert expected.shape == (1, 16) and expected.dtype == np.float32 and expected[0, 0] == 40
    for bad in (None, True, 1.5, 2**31, float("nan")):
        broken = deepcopy(view)
        broken["me"]["stats"][FEATURE_NAMES[0]] = bad
        with pytest.raises(ValueError, match="growth counter"):
            from_views([broken])
    del changed["me"]["stats"][FEATURE_NAMES[0]]
    with pytest.raises(ValueError, match="growth counter"):
        from_views([changed])


@pytest.mark.parametrize("version", [-1, 2, True])
def test_invalid_architecture_version_is_rejected(version):
    with pytest.raises(ValueError, match="version"):
        EntityPolicy(growth_version=version)


def test_native_alias_is_separated_and_zero_projection_preserves_all_old_heads():
    torch.set_num_threads(1)
    torch.manual_seed(151)
    games = pair()
    assert games[0].observe(0) == games[1].observe(0)
    assert games[0].attachment_observation(0) == games[1].attachment_observation(0)
    old = EntityPolicy(width=16, layers=1, heads=2, text_dim=8, attachment_version=1)
    new = EntityPolicy(width=16, layers=1, heads=2, text_dim=8, attachment_version=1, growth_version=1)
    loaded = new.load_state_dict(old.state_dict(), strict=False)
    assert loaded.missing_keys == ["growth_proj.weight"] and not loaded.unexpected_keys
    obs = observations(games, new)
    f, c, m = tensors(obs, "cpu")
    kwargs = policy_kwargs(new, obs, "cpu")
    assert obs["growth"][0, 0] == 0 and obs["growth"][1, 0] == 40
    with torch.no_grad():
        before = old.forward_aux(f, c, m, attachments=kwargs["attachments"])
        after = new.forward_aux(f, c, m, **kwargs)
        assert all(torch.equal(a, b) for a, b in zip(before, after))
    logits, value = new(f, c, m, **kwargs)
    (logits[m].square().mean() + value.square().mean()).backward()
    assert torch.isfinite(new.growth_proj.weight.grad).all()
    assert new.growth_proj.weight.grad.abs().sum() > 0
    with torch.no_grad():
        new.growth_proj.weight[:, 0] = torch.linspace(-1., 1., 16)
        output = new.forward_aux(f, c, m, **kwargs)
        assert any(not torch.equal(x[0], x[1]) for x in output)
    with pytest.raises(ValueError, match="requires aligned"):
        new(f, c, m, attachments=kwargs["attachments"])
    with pytest.raises(ValueError, match="legacy"):
        old(f, c, m, **kwargs)


def test_advice_and_growth_share_one_projection_and_shuffle_with_ppo_rows(tmp_path):
    class Counted:
        def __init__(self, game):
            self.game, self.calls = game, 0
        def strategy_view(self, seat):
            self.calls += 1
            return self.game.strategy_view(seat)
    games = pair()
    wrapped = [Counted(g) for g in games]
    teacher = StrategyTeacher(book())
    obs = observation_rows(games, [(0, 0), (1, 0)], public_attachments=True)
    add_public_context(obs, SimpleNamespace(games=wrapped), [(0, 0), (1, 0)],
                       teacher=teacher, public_growth=True)
    assert [g.calls for g in wrapped] == [1, 1]
    expected = StrategyTeacher(book()).from_env(SimpleNamespace(games=games), [(0, 0), (1, 0)], obs["action_mask"])
    np.testing.assert_array_equal(obs["strategy_bias"], expected)
    actor = EntityPolicy(width=16, layers=1, heads=2, text_dim=8, attachment_version=1, growth_version=1)
    with torch.no_grad():
        actor.growth_proj.weight.normal_(0, .05)
    model = ExpertStrategyPolicy(actor, book()).eval()
    f, c, m = tensors(obs, "cpu")
    dist, value = model.distribution(f, c, m, **policy_kwargs(model, obs, "cpu"))
    actions = dist.probs.argmax(-1)
    order = np.array([1, 0, 1])
    shuffled = {k: v[order] for k, v in obs.items()}
    other, _ = model.distribution(*tensors(shuffled, "cpu"), **policy_kwargs(model, shuffled, "cpu"))
    torch.testing.assert_close(other.log_prob(actions[order]), dist.log_prob(actions)[order], rtol=0, atol=1e-6)
    from training.strategy import digest
    path = tmp_path/"growth.pt"
    torch.save(dict(engine_hash=tg.source_hash(), architecture=model.architecture, model=model.state_dict(),
                    strategy_bundle=book(), strategy_sha256=digest(book())), path)
    loaded, _ = load_checkpoint(path)
    assert loaded.requires_growth and loaded.architecture["growth_version"] == 1
    output = loaded(f, c, m, **policy_kwargs(loaded, obs, "cpu"))
    expected = model(f, c, m, **policy_kwargs(model, obs, "cpu"))
    assert all(torch.equal(a, b) for a, b in zip(output, expected))
    broken = dict(obs, growth=obs["growth"].copy())
    broken["growth"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        policy_kwargs(model, broken, "cpu")


class GrowthEnd(torch.nn.Module):
    requires_growth = True
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, features, cards, masks, growth=None):
        assert growth is not None and growth.shape == (len(features), N_FEATURES)
        assert growth.dtype == torch.float32 and torch.isfinite(growth).all()
        self.calls += 1
        return (torch.zeros_like(masks, dtype=torch.float32).masked_fill(~masks, -1e9),
                growth[:, 0] / 100)


def test_action_validation_does_not_repeat_the_learners_public_projection():
    class Counted:
        def __init__(self, game):
            self.game, self.calls = game, 0
        def __getattr__(self, name):
            return getattr(self.game, name)
        def strategy_view(self, seat):
            self.calls += 1
            return self.game.strategy_view(seat)

    env = LeagueEnv(1, seed=541, opponents=[None], opponent_ids=["bot"], device="cpu",
                    config={"turn_action_caps": [128]}, cancel_guard=True,
                    public_attachments=True, public_growth=True)
    try:
        game = Counted(env.games[0])
        env.games[0] = game
        env.strategy_teacher = StrategyTeacher(book())
        policy_obs = env.observe()
        assert game.calls == 1 and policy_obs["growth"].shape == (1, N_FEATURES)
        assert "strategy_bias" in policy_obs
        counts = dict(env.strategy_teacher.counts)
        validation_obs = env.observe(include_strategy=False, include_growth=False)
        assert game.calls == 1 and "growth" not in validation_obs
        assert "strategy_bias" not in validation_obs
        for key in ("features", "cards", "action_mask", "ready"):
            np.testing.assert_array_equal(validation_obs[key], policy_obs[key])
        action = int(np.flatnonzero(policy_obs["action_mask"][0])[0])
        env.step([action])
        assert game.calls == 1 and dict(env.strategy_teacher.counts) == counts
        # Suppressing advice alone must still supply growth to policy callers.
        next_obs = env.observe(include_strategy=False)
        assert game.calls == 2 and next_obs["growth"].shape == (1, N_FEATURES)
        assert "strategy_bias" not in next_obs
    finally:
        env.close()


def test_league_preserves_growth_at_decision_time_and_restore_rejects_contract_changes():
    model = GrowthEnd().eval()
    kwargs = dict(seed=541, opponents=[model], opponent_ids=["test"], device="cpu",
                  config={"turn_action_caps": [128]}, cancel_guard=True)
    a, b = [LeagueEnv(1, public_growth=True, **kwargs) for _ in range(2)]
    legacy = LeagueEnv(1, **kwargs)
    try:
        a.games[0] = pair()[1]
        a.learners[0] = a.cursors[0] = 0
        b.load(0, a.save(0))
        with pytest.raises(ValueError, match="growth observation version"):
            legacy.load(0, a.save(0))
        with pytest.raises(ValueError, match="growth observation version"):
            a.load(0, legacy.save(0))
        for _ in range(8):
            oa, ob = a.observe(), b.observe()
            np.testing.assert_array_equal(oa["growth"], ob["growth"])
            np.testing.assert_array_equal(oa["growth"], from_views([
                json.loads(a.games[0].strategy_view(a.learners[0]))]))
            action = [int(np.flatnonzero(oa["action_mask"][0])[0])]
            ar, ad, _ = a.step(action)
            br, bd, _ = b.step(action)
            np.testing.assert_array_equal(ar, br)
            np.testing.assert_array_equal(ad, bd)
            assert a.save(0) == b.save(0)
        assert model.calls > 0
    finally:
        a.close(); b.close(); legacy.close()


def test_truncation_bootstraps_from_growth_before_reset(tmp_path):
    from training.common import bootstrap_time_limits
    model = GrowthEnd().eval()
    env = LeagueEnv(1, seed=541, opponents=[model], opponent_ids=["test"], device="cpu",
                    config={"max_turns": 2, "hero_pool": ["TB_BaconShop_HERO_35"]},
                    training_turn_limit=2, truncation_dir=tmp_path, public_growth=True)
    try:
        env.step([30])
        state = json.loads(env.games[0].state_json())
        state["players"][env.learners[0]]["stats"]["eternal_knight_deaths"] = 40
        env.games[0] = tg.Game.from_json(json.dumps(state))
        for _ in range(20):
            action = int(np.flatnonzero(env.observe()["action_mask"][0])[0])
            reward, done, info = env.step([action])
            if done[0]:
                break
        assert info["truncated"][0] and info["final_observation"]["growth"][0, 0] == 40
        assert env.observe()["growth"][0, 0] == 0
        np.testing.assert_allclose(bootstrap_time_limits(model, reward, info, .9, "cpu"), [.36])
    finally:
        env.close()


def test_complete_evaluation_drivers_and_position_actor_accept_growth():
    from training.arena import play_arena
    from training.common import evaluate
    from training.inference import FrozenOpponentGraphs
    from training.position import PublicPositionPolicy
    from test_position import ridge_for
    model = GrowthEnd().eval()
    kwargs = dict(games=8, seed=541, threads=2, device=torch.device("cpu"),
                  config={"turn_action_caps": [128], "hotfix_37_0_1": True},
                  cancel_guard=True, audit_guards=True)
    arena = play_arena([model]+[None]*7, **kwargs)
    assert arena["guards_checked"] and len(arena["lobbies"]) == 8
    before = model.calls
    bots = evaluate(model, **kwargs)
    assert bots["guards_checked"] and model.calls > before
    with pytest.raises(ValueError, match="growth inputs"):
        FrozenOpponentGraphs(model)
    actor = EntityPolicy(width=8, layers=1, heads=2, text_dim=8, growth_version=1).eval()
    encoder = EntityPolicy(width=8, layers=1, heads=2, text_dim=8).eval()
    wrapped = PublicPositionPolicy(actor, encoder, ridge_for(encoder)).eval()
    obs = observations([pair()[1]], actor)
    wrapped.set_context([17])
    with torch.no_grad():
        actual = wrapped(*tensors(obs, "cpu"), **policy_kwargs(wrapped, obs, "cpu"))
        expected = actor(*tensors(obs, "cpu"), **policy_kwargs(actor, obs, "cpu"))
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    with pytest.raises(ValueError, match="legacy observation encoder"):
        PublicPositionPolicy(actor, actor, ridge_for(actor))


def test_optimizer_separates_both_residuals_without_changing_legacy_groups():
    from training.train import make_optimizer, parameter_learning_rate
    actor = EntityPolicy(width=8, layers=1, heads=2, text_dim=8, attachment_version=1, growth_version=1)
    model = ExpertStrategyPolicy(actor, book())
    optimizer = make_optimizer(model, 6e-6, 1e-4, 2e-4)
    assert parameter_learning_rate(optimizer, actor.attachment_proj.weight) == 1e-4
    assert parameter_learning_rate(optimizer, actor.growth_proj.weight) == 2e-4
    ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    old = EntityPolicy(width=8, layers=1, heads=2, text_dim=8, attachment_version=1)
    extra = list(old.attachment_proj.parameters())
    old_ids = {id(p) for p in extra}
    original = torch.optim.Adam([dict(params=[p for p in old.parameters() if id(p) not in old_ids]),
                                 dict(params=extra, lr=1e-4)], lr=6e-6, eps=1e-5)
    assert make_optimizer(old, 6e-6, 1e-4).state_dict() == original.state_dict()


def test_real_growth_ppo_and_full_resume_match_uninterrupted_training(tmp_path):
    from training.strategy import digest
    torch.manual_seed(741)
    knowledge = book()
    knowledge_path = tmp_path/"knowledge.json"
    knowledge_path.write_text(json.dumps(knowledge))
    model = ExpertStrategyPolicy(EntityPolicy(width=16, layers=1, heads=2, text_dim=8,
        attachment_version=1, growth_version=1), knowledge)
    initial = tmp_path/"initial.pt"
    torch.save(dict(engine_hash=tg.source_hash(), training_code_hash="test-only", weights_only=True,
                    architecture=model.architecture, model=model.state_dict(), update=0, steps=0,
                    strategy_bundle=knowledge, strategy_sha256=digest(knowledge),
                    args=dict(seed=741, eval_seed=1741, action_cap=128, hotfix_37_0_1=True)), initial)
    args = [sys.executable, "-m", "training.train", "--device", "cpu", "--envs", "8",
            "--rollout", "128", "--minibatch", "256", "--threads", "1", "--torch-threads", "1",
            "--width", "16", "--layers", "1", "--heads", "2", "--seed", "741",
            "--eval-seed", "1741", "--eval-games", "8", "--eval-every", "100",
            "--warmup-samples", "0", "--opponents", "bot", "--cancel-guard",
            "--hotfix-37-0-1", "--public-growth", "--growth-lr", "0.0001",
            "--public-attachments", "--attachment-lr", "0.0001",
            "--strategy-bundle", str(knowledge_path), "--strategy-coef", "0.02",
            "--league-turn-limit", "100",
            "--skip-baselines", "--checkpoint-seconds", "999999", "--max-hours", "0"]
    for run, updates, start in (("whole", 2, ["--init-from", str(initial)]),
                                ("split", 1, ["--init-from", str(initial)]),
                                ("split", 2, ["--resume", str(tmp_path/"split/latest.pt")])):
        result = subprocess.run(args+["--run-dir", str(tmp_path/run), "--max-updates", str(updates)]+start,
            cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stdout+result.stderr
    a, b = [torch.load(tmp_path/name/"latest.pt", weights_only=False) for name in ("whole", "split")]
    assert a["update"] == b["update"] == 2 and a["steps"] == b["steps"] == 2048
    assert a["model"]["actor.growth_proj.weight"].abs().sum() > 0
    assert a["strategy_sha256"] == b["strategy_sha256"] == digest(knowledge)
    assert all(torch.equal(t, b["model"][name]) for name, t in a["model"].items())
    assert json.loads(gzip.decompress(a["environment"])) == json.loads(gzip.decompress(b["environment"]))
    assert all(json.loads(state)["growth_version"] == 1 for state in json.loads(gzip.decompress(a["environment"])))
    assert torch.equal(a["torch_rng"], b["torch_rng"])
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    for key, values in a["optimizer"]["state"].items():
        assert all(torch.equal(t, b["optimizer"]["state"][key][name]) for name, t in values.items())
