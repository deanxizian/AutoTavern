"""Public information boundaries and lossless batching for optional attachment inputs."""

from copy import deepcopy
from functools import lru_cache
import json

import numpy as np
import pytest
import torch
import taverngym as tg

from training.attachments import AttachmentRows
from training.entity import EntityPolicy


@lru_cache(None)
def card_index(name):
    return next(i for i in range(1, tg.N_CARDS) if tg.card_info(i)[0] == name)


def fixture():
    game = tg.Game(541, json.dumps({"num_players": 8, "record_log": False,
                                   "hotfix_37_0_1": True}))
    for seat in range(8):
        game.step(seat, 30)
    state = json.loads(game.state_json())
    for player in state["players"]:
        for key in ("powers", "pending", "board", "hand", "shop", "trinkets", "effects", "scheduled"):
            player[key] = []
        player.update(skipping=False, gold=30, tier=6)
        for key, value in player["stats"].items():
            if isinstance(value, bool):
                player["stats"][key] = False
            elif isinstance(value, int):
                player["stats"][key] = 0
            elif isinstance(value, list):
                player["stats"][key] = []
    return tg.Game.from_json(json.dumps(state))


def add(game, zone="board", gift="BG36_MidGameEffect_000t74", seat=0):
    game.debug(json.dumps({"cmd": "add_minion", "p": seat, "zone": zone, "card": "BG36_345"}))
    if gift:
        game.debug(json.dumps({"cmd": "edit_minion", "p": seat, "zone": zone, "index": 0,
                               "attach": gift}))


def groups(game, seat=0):
    raw_offsets, raw_cards = game.attachment_observation(seat)
    offsets, cards = np.frombuffer(raw_offsets, "<i4"), np.frombuffer(raw_cards, "<i4")
    assert len(offsets) == tg.OBS_CARDS + 1 and offsets[0] == 0 and offsets[-1] == len(cards)
    return [cards[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]


def slots():
    layout = dict(tg.layout())
    prefix = 1 + layout["N_POWERS"] + 2 + 3 * layout["N_QUESTS"] + layout["N_SECRETS"]
    return prefix, prefix + 7, prefix + 17, prefix + 25, tg.OBS_CARDS - 49


def test_own_slot_alignment_empty_groups_and_no_state_or_rng_mutation():
    game = fixture()
    for zone, gift in (("board", "74"), ("hand", "75"), ("shop", "51")):
        add(game, zone, "BG36_MidGameEffect_000t" + gift)
    before, observation, mask = game.state_json(), game.observe(0), game.legal_mask(0)
    result = groups(game)
    board, hand, shop, _, _ = slots()
    assert result[board] == [card_index("BG36_MidGameEffect_000t74")]
    assert result[hand] == [card_index("BG36_MidGameEffect_000t75")]
    assert result[shop] == [card_index("BG36_MidGameEffect_000t51")]
    assert all(not value for i, value in enumerate(result) if i not in (board, hand, shop))
    assert game.state_json() == before and game.observe(0) == observation and game.legal_mask(0) == mask


def test_private_opponents_and_rng_are_invisible_but_scouted_attachments_are_visible():
    game = fixture()
    add(game, seat=1)
    baseline = groups(game)
    state = json.loads(game.state_json())
    remembered = deepcopy(state["players"][1]["board"])
    state["players"][1]["board"][0]["attached"] = [card_index("BG36_MidGameEffect_000t75") - 1]
    state["players"][1]["seen_board"] = deepcopy(state["players"][1]["board"])
    state["rng"] = json.loads(tg.Game(999).state_json())["rng"]
    changed = tg.Game.from_json(json.dumps(state))
    add(changed, zone="hand", seat=1)
    add(changed, zone="shop", seat=1)
    assert groups(changed) == baseline
    state = json.loads(changed.state_json())
    for viewer in (0, 7):
        state["players"][viewer]["scouted"] = [{"board": [], "turn": 0} for _ in range(8)]
    state["players"][0]["scouted"][1] = {"board": remembered, "turn": 1}
    observed = tg.Game.from_json(json.dumps(state))
    expected = deepcopy(baseline)
    expected[slots()[-1]] = [card_index("BG36_MidGameEffect_000t74")]
    assert groups(observed) == expected
    # Rotate the viewer: player7 sees player1 in its second opponent group.
    state["players"][7]["scouted"][1] = {"board": remembered, "turn": 1}
    observed = tg.Game.from_json(json.dumps(state))
    assert groups(observed, 7)[slots()[-1] + 7] == expected[slots()[-1]]


@pytest.mark.parametrize("kind", ["Discover", "Timewarp", "Target"])
def test_pending_options_use_the_same_visible_entity_order(kind):
    game = fixture()
    for zone, gift in (("board", "74"), ("hand", "75"), ("shop", "51")):
        add(game, zone, "BG36_MidGameEffect_000t" + gift)
    state = json.loads(game.state_json())
    own = state["players"][0]
    items = [deepcopy(own["shop"][0]["item"]), {"Minion": deepcopy(own["board"][0])},
             deepcopy(own["hand"][0]["item"])]
    if kind == "Discover":
        pending = {kind: {"source": own["board"][0]["card"], "options": items}}
    elif kind == "Timewarp":
        pending = {kind: {"options": [[item, 1] for item in items], "chronum": 4, "major": False}}
    else:
        pending = {kind: {"options": [own["shop"][0]["sid"], own["board"][0]["eid"], own["hand"][0]["hid"]],
                          "cont": {"HeroPower": {"slot": 0, "opt": 0}}}}
    own["pending"] = [pending]
    game = tg.Game.from_json(json.dumps(state))
    result = groups(game)
    choices = slots()[3]
    assert result[choices:choices + 3] == [[card_index("BG36_MidGameEffect_000t" + n)]
                                          for n in ("51", "74", "75")]
    assert not result[choices + 3]  # Includes Timewarp's non-card leave option.


def test_repeated_attachments_are_never_silently_truncated():
    game = fixture()
    add(game)
    state = json.loads(game.state_json())
    state["players"][0]["board"][0]["attached"] *= 1025
    game = tg.Game.from_json(json.dumps(state))
    rows = AttachmentRows.from_games([game], [(0, 0)])
    assert rows.counts[0, slots()[0]] == 1025 and len(rows.cards) == 1025
    assert groups(game)[slots()[0]] == [card_index("BG36_MidGameEffect_000t74")] * 1025


def test_privileged_debug_view_and_invalid_seat_are_refused():
    game = fixture()
    with pytest.raises(ValueError, match="seat"):
        game.attachment_observation(8)
    state = json.loads(game.state_json())
    state["players"][0]["sees_all_boards"] = True
    with pytest.raises(ValueError, match="privileged"):
        tg.Game.from_json(json.dumps(state)).attachment_observation(0)


def test_ragged_batch_selection_reordering_duplicates_and_empty_rows():
    a, b = fixture(), fixture()
    add(a)
    add(b, "hand", "BG36_MidGameEffect_000t75")
    rows = AttachmentRows.from_games([a, b, fixture()], [(0, 0), (1, 0), (2, 0)])
    selected = rows[[2, 1, 1, 0]]
    assert np.array_equal(selected.counts, rows.counts[[2, 1, 1, 0]])
    assert selected.cards.tolist() == [card_index("BG36_MidGameEffect_000t75")] * 2 + rows.cards[:1].tolist()
    joined = AttachmentRows.concatenate([rows[:1], rows[1:]])
    assert np.array_equal(joined.counts, rows.counts) and np.array_equal(joined.cards, rows.cards)
    tensors = selected.tensors("cpu")
    assert len(tensors.offsets) == 4 * tg.OBS_CARDS + 1 and tensors.offsets[-1].item() == 3
    assert len(rows[[]]) == 0 and len(AttachmentRows.concatenate([])) == 0


@pytest.mark.parametrize("failure", ["negative", "overflow", "float", "zero_id", "bad_id", "mismatch"])
def test_malformed_rows_fail_closed(failure):
    counts, cards = np.zeros((1, tg.OBS_CARDS), np.int64), np.array([1], np.int64)
    counts[0, 0] = 1
    if failure == "negative": counts[0, 0] = -1
    if failure == "overflow": counts[0, 0] = 1 << 32
    if failure == "float": counts = counts.astype(float)
    if failure == "zero_id": cards[0] = 0
    if failure == "bad_id": cards[0] = tg.N_CARDS
    if failure == "mismatch": counts[0, 0] = 2
    with pytest.raises(ValueError):
        AttachmentRows(counts, cards)


def alias_pair():
    a, b = fixture(), fixture()
    for game, gift in ((a, "74"), (b, "75")):
        add(game, gift="BG36_MidGameEffect_000t" + gift)
        add(game, "hand", gift=None)
    assert a.observe(0) == b.observe(0) and a.legal_mask(0) == b.legal_mask(0)
    assert groups(a) != groups(b)
    return a, b


def test_identical_legacy_inputs_can_have_different_own_play_consequences():
    a, b = alias_pair()
    a.step(0, 47)
    b.step(0, 47)
    left, right = (json.loads(game.state_json())["players"][0]["board"][0] for game in (a, b))
    assert (left["atk"], left["max_hp"]) == (6, 3)
    assert (right["atk"], right["max_hp"]) == (3, 6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_zero_residual_preserves_policy_and_receives_finite_nonzero_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    a, b = alias_pair()
    raw_f, raw_c = a.observe(0)
    features = torch.tensor(np.frombuffer(raw_f, "<f4").copy()[None], device=device)
    cards = torch.tensor(np.frombuffer(raw_c, "<i4").copy()[None], device=device)
    masks = torch.tensor(np.frombuffer(a.legal_mask(0), np.uint8).copy()[None].astype(bool), device=device)
    old = EntityPolicy(width=32, layers=1, heads=4, text_dim=8).to(device)
    new = EntityPolicy(width=32, layers=1, heads=4, text_dim=8, attachment_version=1).to(device)
    loaded = new.load_state_dict(old.state_dict(), strict=False)
    assert loaded.missing_keys == ["attachment_proj.weight"] and not loaded.unexpected_keys
    extra_a, extra_b = (AttachmentRows.from_games([game], [(0, 0)]).tensors(device) for game in (a, b))
    with torch.no_grad():
        expected = old.forward_aux(features, cards, masks)
        actual = new.forward_aux(features, cards, masks, extra_a)
        assert all(torch.equal(x, y) for x, y in zip(expected, actual))
        old_dist, old_value, old_aux = old.distribution(features, cards, masks, auxiliary=True)
        new_dist, new_value, new_aux = new.distribution(features, cards, masks, auxiliary=True,
                                                       attachments=extra_a)
        assert torch.equal(old_dist.probs, new_dist.probs)
        assert torch.equal(old_value, new_value) and torch.equal(old_aux, new_aux)
        empty = AttachmentRows.empty(1).tensors(device)
        assert all(torch.equal(x, y) for x, y in zip(expected, new.forward_aux(features, cards, masks, empty)))
    with pytest.raises(ValueError, match="requires public"):
        new(features, cards, masks)
    with pytest.raises(ValueError, match="legacy"):
        old(features, cards, masks, extra_a)
    logits, value = new(features, cards, masks, extra_a)
    (logits[masks].square().mean() + value.square().mean()).backward()
    gradient = new.attachment_proj.weight.grad
    assert torch.isfinite(gradient).all() and gradient.abs().sum().item() > 0
    with torch.no_grad():
        new.attachment_proj.weight.copy_(torch.eye(32, device=device))
        left, right = new(features, cards, masks, extra_a), new(features, cards, masks, extra_b)
        assert any(not torch.equal(x, y) for x, y in zip(left, right))


@pytest.mark.parametrize("version", [-1, 2, True])
def test_unsupported_attachment_model_versions_fail_closed(version):
    with pytest.raises(ValueError, match="version"):
        EntityPolicy(attachment_version=version)


def assert_same_rows(left, right):
    np.testing.assert_array_equal(left.counts, right.counts)
    np.testing.assert_array_equal(left.cards, right.cards)


class AttachmentEndTurn(torch.nn.Module):
    requires_attachments = True

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.seen = set()

    def forward(self, features, cards, masks, attachments=None):
        assert attachments is not None
        assert len(attachments.offsets) == len(features) * tg.OBS_CARDS + 1
        assert int(attachments.offsets[-1]) == len(attachments.cards)
        self.calls += 1
        self.seen.update(attachments.cards.cpu().tolist())
        return (torch.zeros_like(masks, dtype=torch.float32).masked_fill(~masks, -1e9),
                torch.full((len(features),), .25, device=features.device))


def test_league_collects_at_decision_time_and_restores_public_rows():
    from training.league import LeagueEnv
    model = AttachmentEndTurn().eval()
    kwargs = dict(seed=541, opponents=[model], opponent_ids=["test"], device="cpu",
                  config={"turn_action_caps": [128]}, cancel_guard=True)
    a, b = [LeagueEnv(2, public_attachments=True, **kwargs) for _ in range(2)]
    legacy = LeagueEnv(2, **kwargs)
    try:
        for lane in range(2):
            a.games[lane] = fixture()
            for seat in range(8):
                add(a.games[lane], "hand", "BG36_MidGameEffect_000t" + ("74" if seat % 2 else "75"), seat)
            a.cursors[lane] = a.learners[lane]
            b.load(lane, a.save(lane))
        with pytest.raises(ValueError, match="observation version"):
            legacy.load(0, a.save(0))
        with pytest.raises(ValueError, match="observation version"):
            a.load(0, legacy.save(0))
        for _ in range(16):
            oa, ob = a.observe(), b.observe()
            assert_same_rows(oa["attachments"], ob["attachments"])
            assert_same_rows(oa["attachments"], AttachmentRows.from_games(
                a.games, list(enumerate(a.learners))))
            actions = [int(np.flatnonzero(mask)[0]) for mask in oa["action_mask"]]
            ar, ad, _ = a.step(actions)
            br, bd, _ = b.step(actions)
            np.testing.assert_array_equal(ar, br)
            np.testing.assert_array_equal(ad, bd)
            assert [a.save(i) for i in range(2)] == [b.save(i) for i in range(2)]
        assert model.calls and {card_index("BG36_MidGameEffect_000t" + n) for n in ("74", "75")} <= model.seen
    finally:
        a.close(); b.close(); legacy.close()


def test_time_limit_preserves_final_attachments_before_reset(tmp_path):
    from training.common import bootstrap_time_limits
    from training.league import LeagueEnv
    model = AttachmentEndTurn().eval()
    env = LeagueEnv(1, seed=541, opponents=[model], opponent_ids=["test"], device="cpu",
                    config={"max_turns": 2, "hero_pool": ["TB_BaconShop_HERO_35"]},
                    training_turn_limit=2, truncation_dir=tmp_path, public_attachments=True)
    try:
        env.step([30])
        add(env.games[0], "hand", seat=env.learners[0])
        for _ in range(20):
            action = int(np.flatnonzero(env.observe()["action_mask"][0])[0])
            reward, done, info = env.step([action])
            if done[0]:
                break
        else:
            pytest.fail("Expected the external training horizon")
        assert info["truncated"][0] and info["placement"][0] == 0
        evidence = json.loads(next(tmp_path.glob("*.json")).read_text())["environment"]
        original = tg.Game.from_json(evidence["game"])
        expected = AttachmentRows.from_games([original], [(0, evidence["learner"])])
        assert len(expected.cards) > 0
        assert_same_rows(info["final_observation"]["attachments"], expected)
        assert not len(env.observe()["attachments"].cards)
        np.testing.assert_allclose(bootstrap_time_limits(model, reward, info, .9, "cpu"), [.225])
    finally:
        env.close()


def test_ragged_opponents_bypass_fixed_cuda_graph_cache():
    from training.inference import FrozenOpponentGraphs
    from training.league import LeagueEnv
    model = AttachmentEndTurn().eval().requires_grad_(False)
    with pytest.raises(ValueError, match="eager inference"):
        FrozenOpponentGraphs(model, 2)
    if not torch.cuda.is_available():
        pytest.skip("CUDA league graph selection needs GPU")
    env = LeagueEnv(1, seed=541, opponents=[model, model], opponent_ids=["a", "a"],
                    device="cuda", opponent_graph_max_batch=2, public_attachments=True)
    try:
        assert env.opponents[0] is model and env.opponents[1] is model
    finally:
        env.close()


def test_both_evaluation_drivers_pass_public_inputs_through_complete_lobbies():
    from training.arena import play_arena
    from training.common import evaluate
    torch.set_num_threads(2)
    model = AttachmentEndTurn().eval()
    kwargs = dict(games=8, seed=541, threads=2, device=torch.device("cpu"),
                  config={"turn_action_caps": [128], "hotfix_37_0_1": True},
                  cancel_guard=True, audit_guards=True)
    arena = play_arena([model] + [None] * 7, **kwargs)
    assert len(arena["lobbies"]) == 8 and arena["guards_checked"]
    before = model.calls
    bots = evaluate(model, **kwargs)
    assert sum(len(v) for v in bots["placements_by_seat"].values()) == 8
    assert bots["guards_checked"] and model.calls > before


def test_position_decoder_passes_extras_only_to_actor():
    from training.common import tensors
    from training.league import observation_rows
    from training.position import PublicPositionPolicy
    from test_position import ridge_for
    actor = EntityPolicy(width=8, layers=1, heads=2, text_dim=8, attachment_version=1).eval()
    encoder = EntityPolicy(width=8, layers=1, heads=2, text_dim=8).eval()
    policy = PublicPositionPolicy(actor, encoder, ridge_for(encoder)).eval()
    game = fixture()
    add(game)
    obs = observation_rows([game], [(0, 0)], public_attachments=True)
    inputs, extras = tensors(obs, "cpu"), obs["attachments"].tensors("cpu")
    policy.set_context([17])
    with torch.no_grad():
        expected = actor(*inputs, attachments=extras)
        actual = policy(*inputs, attachments=extras)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    with pytest.raises(ValueError, match="legacy observation encoder"):
        PublicPositionPolicy(actor, actor, ridge_for(actor))


def test_real_ppo_updates_attachment_weights_and_resumes_exactly(tmp_path):
    import gzip
    import subprocess
    import sys
    from pathlib import Path
    from training.common import load_checkpoint

    torch.manual_seed(741)
    model = EntityPolicy(width=16, layers=1, heads=2, text_dim=8, attachment_version=1)
    initial = tmp_path / "initial.pt"
    torch.save(dict(engine_hash=tg.source_hash(), training_code_hash="test-only", weights_only=True,
                    architecture=model.architecture, model=model.state_dict(), update=0, steps=0,
                    args=dict(seed=741, eval_seed=1741, action_cap=128, hotfix_37_0_1=True)), initial)
    base = [sys.executable, "-m", "training.train", "--device", "cpu", "--envs", "8",
            "--rollout", "256", "--minibatch", "256", "--threads", "2", "--torch-threads", "2",
            "--width", "16", "--layers", "1", "--heads", "2", "--seed", "741",
            "--eval-seed", "1741", "--eval-games", "8", "--eval-every", "100",
            "--warmup-samples", "0", "--opponents", "bot", "--cancel-guard",
            "--hotfix-37-0-1", "--public-attachments", "--league-turn-limit", "100",
            "--attachment-lr", "0.0001",
            "--skip-baselines", "--checkpoint-seconds", "999999", "--max-hours", "0"]
    for run, updates, start in (("whole", 2, ["--init-from", str(initial)]),
                                 ("split", 1, ["--init-from", str(initial)]),
                                 ("split", 2, ["--resume", str(tmp_path/"split/latest.pt")])):
        result = subprocess.run(base + ["--run-dir", str(tmp_path/run), "--max-updates", str(updates)] + start,
                                cwd=Path(__file__).resolve().parents[1], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (tmp_path/f"{run}-{updates}.log").write_text(result.stdout)
        assert result.returncode == 0, result.stdout[-6000:]
    _, whole = load_checkpoint(tmp_path/"whole/latest.pt", "cpu")
    _, split = load_checkpoint(tmp_path/"split/latest.pt", "cpu")
    assert whole["update"] == split["update"] == 2
    assert whole["steps"] == split["steps"] == 4096
    assert [g["lr"] for g in whole["optimizer"]["param_groups"]] == [0.00002, 0.0001]
    assert whole["optimizer"]["param_groups"] == split["optimizer"]["param_groups"]
    assert torch.count_nonzero(whole["model"]["attachment_proj.weight"]) > 0
    for key in whole["model"]:
        assert torch.equal(whole["model"][key], split["model"][key]), key
    for key, state in whole["optimizer"]["state"].items():
        for name, tensor in state.items():
            assert torch.equal(tensor, split["optimizer"]["state"][key][name]), (key, name)
    assert torch.equal(whole["torch_rng"], split["torch_rng"])
    assert json.loads(gzip.decompress(whole["environment"])) == json.loads(gzip.decompress(split["environment"]))
    metrics = [json.loads(line) for line in (tmp_path/"whole/metrics.jsonl").read_text().splitlines()]
    assert sum(row.get("attachment_occurrences", 0) for row in metrics) > 0
