"""Human-cohort hero preference is optional, offer-relative and never an action filter."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import taverngym as tg
from stable_baselines3.common.callbacks import BaseCallback
from taverngym.learning import rewards
from taverngym.learning.actor_sampling import ActorOutput, UnifiedActorSampler
from taverngym.learning.experiment import log_evaluation
from taverngym.learning.model import ModelConfig, ObservationLayout
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.tracking import WandbTracker
from taverngym.learning.vector_env import NativeVectorEnv


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "configs/hero-selection-priors-hsreplay-20261029.json").read_text())
MEANS = {row["blizzard_id"]: row["mean_placement"]["top_1_percent"] for row in DATA["heroes"]}
IDS = rewards._active_hero_indices()
LAYOUT = ObservationLayout()
NATIVE = LAYOUT.native
PENDING = NATIVE["GLOBAL_F"] + NATIVE["N_QUESTS"] * NATIVE["QUEST_F"]
GOOD, WORST = "TB_BaconShop_HERO_10", "TB_BaconShop_HERO_34"
MIDDLE = "TB_BaconShop_HERO_37"
PRIOR = {"operation_cost": 0, "cycle_cost": 0, "hero_selection_bonus_scale": .01,
         "hero_selection_bonus_cap": .005, "hero_selection_mean_placements": MEANS}


@pytest.fixture(autouse=True)
def no_adam(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Hero-prior preparation must not update Adam")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


def offered(names, pending=1, turn=0):
    features = np.zeros(tg.OBS_FLOATS, np.float32)
    features[0], features[1], features[4] = turn / 20, turn == 0, 1
    features[PENDING + pending] = 1
    cards = np.zeros(tg.OBS_CARDS, np.int32)
    cards[LAYOUT.cards["choices"]][:len(names)] = [IDS[name] for name in names]
    mask = np.zeros(tg.N_ACTIONS, bool)
    mask[NATIVE["A_CHOOSE"]:NATIVE["A_CHOOSE"] + len(names)] = True
    return {"features": features, "cards": cards, "action_mask": mask}


def selected(before, name, *, still_hero_phase=False):
    after = {key: value.copy() for key, value in before.items()}
    after["features"][PENDING:PENDING + NATIVE["PENDING_KINDS"]] = 0
    after["features"][PENDING] = 1
    after["features"][1] = still_hero_phase
    after["features"][0] = 0 if still_hero_phase else .05
    after["cards"][0] = IDS[name]
    return after


def tracker(**kwargs):
    return rewards.ActionRewardShaper(rewards.RewardShapingConfig(**{**PRIOR, **kwargs}))


def pick(shaper, names, name, **kwargs):
    before = offered(names)
    return shaper.step(NATIVE["A_CHOOSE"] + names.index(name), before, selected(before, name),
                       0, **kwargs)


def test_only_twelve_numeric_rows_are_preserved_without_cohort_mixing():
    raw = (ROOT / DATA["source"]["path"]).read_bytes()
    assert len(raw) == DATA["source"]["bytes"] == 26300
    assert hashlib.sha256(raw).hexdigest() == DATA["source"]["sha256"]
    assert DATA["default_cohort"] == "top_1_percent"
    rows = {row["name_zh"]: row for row in DATA["heroes"]}
    matched = 0
    for line in raw.decode().splitlines():
        cells = line.split("\t")
        if len(cells) == 5 and cells[0] in rows:
            row = rows[cells[0]]
            assert row["mean_placement"]["top_50_percent"] == float(cells[1])
            assert row["mean_placement"]["top_1_percent"] == float(cells[2])
            assert row["hero_sample_count"] is None
            assert row["blizzard_id"] in IDS and "Tutorial" not in row["blizzard_id"]
            matched += 1
    assert matched == len(MEANS) == len(rows) == 12


@pytest.mark.parametrize("names", [[GOOD, MIDDLE, WORST], [WORST, GOOD, MIDDLE],
                                   [MIDDLE, WORST, GOOD]])
def test_offer_order_does_not_change_relative_bonus(names):
    shaper = tracker()
    reward, _ = pick(shaper, names, GOOD)
    assert reward == pytest.approx(.0034)
    assert shaper._totals["hero_selection_bonus_count"] == 1
    assert shaper._totals["hero_selection_count"] == 1
    assert shaper._totals["hero_selection_scorable"] == 1
    assert shaper._totals["hero_selection_known_fraction"] == 1
    assert shaper._totals["early_actions"] == 0


def test_worst_tied_or_single_candidate_gives_zero():
    shaper = tracker()
    assert pick(shaper, [GOOD, WORST], WORST)[0] == 0
    tied = ["TB_BaconShop_HERO_08", "BG34_HERO_002", "TB_BaconShop_HERO_68"]
    shaper = tracker()
    assert pick(shaper, tied, tied[0])[0] == 0
    assert shaper._totals["hero_selection_scorable"] == 1
    for names in ([GOOD], [GOOD, GOOD]):
        shaper = tracker()
        assert pick(shaper, names, GOOD)[0] == 0
        assert shaper._totals["hero_selection_scorable"] == 0
        assert shaper._totals["hero_selection_offered_heroes"] == 1


@pytest.mark.parametrize("choose_unknown", [False, True])
def test_any_unknown_hero_makes_entire_offer_unscorable(choose_unknown):
    unknown = next(name for name in IDS if name not in MEANS)
    shaper = tracker()
    assert pick(shaper, [GOOD, unknown], unknown if choose_unknown else GOOD)[0] == 0
    assert shaper._totals["hero_selection_unknown_offer"] == 1
    assert shaper._totals["hero_selection_unknown_heroes"] == 1
    assert shaper._totals["hero_selection_known_fraction"] == .5
    assert shaper._totals["hero_selection_scorable"] == 0


def test_only_legal_offered_choices_enter_coverage():
    unknown = next(name for name in IDS if name not in MEANS)
    before = offered([GOOD, WORST, unknown])
    before["action_mask"][NATIVE["A_CHOOSE"] + 2] = False
    shaper = tracker()
    assert shaper.step(NATIVE["A_CHOOSE"], before, selected(before, GOOD), 0)[0] == pytest.approx(
        .0034)
    assert shaper._totals["hero_selection_unknown_heroes"] == 0


def test_actual_bonus_cap_and_per_episode_flag_survive_cycle_clear():
    shaper = tracker(hero_selection_bonus_scale=.1)
    assert pick(shaper, [GOOD, WORST], GOOD)[0] == .005
    shaper._clear_history()
    assert pick(shaper, [GOOD, WORST], GOOD)[0] == 0
    assert shaper._totals["hero_selection_count"] == 1
    totals = shaper.finish_terminal(1, 1)
    assert totals["hero_selection_bonus"] == .005 and totals["hero_selection_bonus_count"] == 1
    assert pick(shaper, [GOOD, WORST], GOOD)[0] == .005
    shaper.reset()
    assert pick(shaper, [GOOD, WORST], GOOD)[0] == .005


@pytest.mark.parametrize("case", ["discover", "power", "recruit", "illegal", "wrong_hero",
                                 "pending", "terminal", "gameover", "intercepted"])
def test_non_hero_or_uncompleted_selection_does_not_reward(case):
    shaper = tracker()
    before = offered([GOOD, WORST], pending={"discover": 2, "power": 5}.get(case, 1),
                     turn=1 if case == "recruit" else 0)
    after = selected(before, WORST if case == "wrong_hero" else GOOD)
    if case == "illegal":
        before["action_mask"][NATIVE["A_CHOOSE"]] = False
    if case == "pending":
        after["features"][PENDING + 1] = 1
    if case == "gameover":
        after["features"][2] = 1
    reward, totals = shaper.step(NATIVE["A_CHOOSE"], before, after, 0,
        done=case in ("terminal", "intercepted"),
        terminal_reason="action_limit_forfeit" if case == "intercepted" else None)
    assert reward == 0
    assert (totals or shaper._totals)["hero_selection_count"] == 0


def test_selection_succeeds_while_other_seats_remain_in_hero_phase():
    before = offered([GOOD, WORST])
    shaper = tracker()
    assert shaper.step(NATIVE["A_CHOOSE"], before,
                       selected(before, GOOD, still_hero_phase=True), 0)[0] == pytest.approx(.0034)


@pytest.mark.parametrize("name", ["not-a-real-hero", "TB_BaconShop_HERO_10_Tutorial", "BG28_897"])
def test_enabled_unknown_inactive_or_nonhero_configuration_fails_before_sampling(name):
    with pytest.raises(ValueError, match="unknown or non-active hero"):
        tracker(hero_selection_mean_placements={name: 4.})


@pytest.mark.parametrize("means", [[], {GOOD: None}, {GOOD: float("nan")},
                                   {GOOD: 0}, {GOOD: 9}, {GOOD: "3.04"}])
def test_invalid_mean_placement_table_fails(means):
    with pytest.raises(ValueError, match="Hero selection means"):
        tracker(hero_selection_mean_placements=means)


@pytest.mark.parametrize("field", ["hero_selection_bonus_scale", "hero_selection_bonus_cap"])
def test_invalid_hero_bonus_magnitude_fails(field):
    for value in (-.001, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            tracker(**{field: value})


def test_catalog_mapping_uses_cid_plus_one_and_is_cached(monkeypatch):
    catalog = tg.catalog_json()
    calls = []
    original = tg.catalog_json
    rewards._active_hero_indices.cache_clear()
    monkeypatch.setattr(tg, "catalog_json", lambda: calls.append(1) or catalog)
    try:
        for _ in range(3):
            shaper = tracker()
            for row in json.loads(catalog):
                if row["id"] in MEANS:
                    assert shaper._hero_means[row["cid"] + 1] == MEANS[row["id"]]
        assert len(calls) == 1
    finally:
        rewards._active_hero_indices.cache_clear()
        monkeypatch.setattr(tg, "catalog_json", original)


def test_zero_fields_normalize_only_new_defaults():
    old = {"operation_cost": 0, "cycle_cost": .01}
    defaults = {"hero_selection_bonus_scale": 0, "hero_selection_bonus_cap": .005,
                "hero_selection_mean_placements": {}}
    assert rewards.reward_protocols_equal(old, {**old, **defaults})
    assert not rewards.reward_protocols_equal(old, {**old, "hero_selection_bonus_scale": .01})
    assert not rewards.reward_protocols_equal(old, {**old, "hero_selection_mean_placements": MEANS})
    assert not rewards.reward_protocols_equal(old, {**old, "cycle_forfeit_repetitions": None})
    assert not rewards.reward_protocols_equal(None, {})


NATIVE_CONFIG = {"hero_pool": [GOOD, MIDDLE, WORST], "hero_choices": 3,
                 "turn_action_caps": [3]}


def best_actions(obs):
    masks = obs["action_mask"]
    actions = masks.argmax(axis=1)
    for i in range(len(actions)):
        if obs["features"][i, 1] and obs["features"][i, PENDING + 1]:
            offered_cards = obs["cards"][i, LAYOUT.cards["choices"]]
            options = [j for j in range(NATIVE["N_CHOICE"])
                       if masks[i, NATIVE["A_CHOOSE"] + j]]
            index = min(options, key=lambda j: MEANS[next(
                name for name in MEANS if IDS[name] == offered_cards[j])])
            actions[i] = NATIVE["A_CHOOSE"] + index
    return actions


def expected_bonus(before, action):
    offered_cards = before["cards"][0, LAYOUT.cards["choices"]]
    scores = {IDS[name]: value for name, value in MEANS.items()}
    means = [scores[int(hero)] for hero in offered_cards if hero]
    chosen = scores[int(offered_cards[int(action) - NATIVE["A_CHOOSE"]])]
    return min(.005, .01 * (max(means) - chosen))


def test_native_reward_changes_only_return_not_state_legality_or_placement():
    base = {"operation_cost": 0, "cycle_cost": 0}
    raw, shaped = [NativeVectorEnv(1, seed=13, threads=1, config=NATIVE_CONFIG,
                  reward_shaping=costs) for costs in (base, PRIOR)]
    a, b = raw.reset(), shaped.reset()
    bonus = expected_bonus(b, best_actions(b)[0])
    completed = 0
    try:
        for step in range(80):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            actions = best_actions(a)
            a, ar, ad, ai = raw.step(actions)
            b, br, bd, bi = shaped.step(actions)
            assert raw.batch.save(0) == shaped.batch.save(0)
            np.testing.assert_array_equal(ad, bd)
            # A reset begins a fresh hero phase, so each actual selection may earn a bonus.
            if ar[0] == br[0]:
                assert ad[0] or not raw._obs["features"][0, PENDING + 1]
            else:
                assert br[0] - ar[0] == pytest.approx(bonus)
            if ad[0]:
                completed += 1
                assert ai[0]["placement"] == bi[0]["placement"]
                assert bi[0]["reward_components"]["hero_selection_bonus_count"] == 1
                assert bi[0]["reward_components"]["early_actions"] > 0
                break
        assert completed == 1
    finally:
        raw.close()
        shaped.close()


def test_zero_scale_explicit_defaults_preserve_native_reward_and_trajectory():
    legacy = {"operation_cost": 0, "cycle_cost": 0}
    explicit = {**legacy, "hero_selection_bonus_scale": 0, "hero_selection_bonus_cap": .005,
                "hero_selection_mean_placements": {}}
    a, b = [NativeVectorEnv(1, seed=13, config=NATIVE_CONFIG, reward_shaping=c)
            for c in (legacy, explicit)]
    left, right = a.reset(), b.reset()
    try:
        ended = 0
        for _ in range(80):
            actions = best_actions(left)
            left, lr, ld, li = a.step(actions)
            right, rr, rd, ri = b.step(actions)
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
            np.testing.assert_array_equal(lr, rr)
            np.testing.assert_array_equal(ld, rd)
            assert a.batch.save(0) == b.batch.save(0)
            if ld[0]:
                ended += 1
                assert li[0]["placement"] == ri[0]["placement"]
                break
        assert ended == 1
    finally:
        a.close()
        b.close()


def test_hero_bonus_reaches_gae_without_optimizer_updates(monkeypatch):
    torch.set_num_threads(2)
    env = NativeVectorEnv(1, seed=13, threads=1, config=NATIVE_CONFIG, reward_shaping=PRIOR)
    model = EntityPPO(EntityPolicy, env, n_steps=2, batch_size=2, device="cpu", seed=13,
        gamma=1, gae_lambda=1, n_epochs=1, policy_kwargs={"model_config": asdict(
            ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})

    def forward(obs, deterministic=False, action_masks=None):
        arrays = {key: value.detach().cpu().numpy() for key, value in obs.items()}
        actions = torch.from_numpy(best_actions(arrays))
        return actions, torch.zeros((1, 1)), torch.zeros(1)

    class Continue(BaseCallback):
        def _on_step(self):
            return True

    monkeypatch.setattr(model.policy, "forward", forward)
    monkeypatch.setattr(model.policy, "predict_values", lambda obs: torch.zeros((1, 1)))
    before_weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
    try:
        _, callback = model._setup_learn(2, Continue())
        wanted = expected_bonus(model._last_obs, best_actions(model._last_obs)[0])
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 2)
        np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0], [wanted, 0], atol=1e-8)
        np.testing.assert_allclose(model.rollout_buffer.returns[:, 0], [wanted, 0], atol=1e-8)
        np.testing.assert_allclose(model.rollout_buffer.advantages[:, 0], [wanted, 0], atol=1e-8)
        assert not model.policy.optimizer.state and model.optimizer_step_count() == 0
        assert all(torch.equal(value, before_weights[key])
                   for key, value in model.policy.state_dict().items())
    finally:
        env.close()


def test_unified_actor_awards_initial_choice_once_across_collect_calls():
    class Policy:
        def act(self, obs):
            actions = best_actions(obs)
            return ActorOutput(actions, np.zeros(len(actions), np.float32),
                               np.zeros(len(actions), np.float32))

    sampler = UnifiedActorSampler(1, Policy(), Policy(), frozen_opponents=1,
        seed=13, threads=1, config=NATIVE_CONFIG, reward_shaping=PRIOR)
    before = sampler.learner_observation()
    wanted = expected_bonus(before, best_actions(before)[0])
    transitions = []
    sampler.collect(1, transitions.append)
    ledger = sampler.shapers[0]
    sampler.collect(1, transitions.append)
    assert [t.reward for t in transitions] == pytest.approx([wanted, 0])
    assert sampler.shapers[0] is ledger
    assert ledger._totals["hero_selection_count"] == 1
    assert ledger._totals["hero_selection_bonus_count"] == 1


def test_new_hero_metrics_flow_through_mock_train_val_greedy_journals(tmp_path, monkeypatch):
    recorded = []
    run = SimpleNamespace(entity="eric_eric_", project="AutoTavern", id="mock", url="mock",
        define_metric=lambda *a, **k: None, log=recorded.append, finish=lambda **k: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **k: run,
                                                             Settings=lambda **k: k))
    tracker = WandbTracker(tmp_path, {"tracking": {"mode": "offline"}})
    keys = [key for key in rewards.REWARD_METRICS if key.startswith("hero_selection_")]
    metrics = {key: 1. for key in keys}
    metrics.update(mean_placement=3., first_rate=.2, top4_rate=.7)
    try:
        tracker.log({f"rollout/{key}": value for key, value in metrics.items()}, 0)
        values = {}
        log_evaluation(SimpleNamespace(record=lambda key, value: values.update({key: value})),
            "val", {"conditions": {"mixed_sampled": metrics, "mixed_greedy": metrics}})
        tracker.log(values, 0)
    finally:
        tracker.finish()
    rows = [json.loads(line) for line in
            (tmp_path / "wandb-metrics.jsonl").read_text().splitlines()]
    assert rows == recorded
    for key in keys:
        assert f"train/{key}" in rows[0]
        assert f"val/{key}" in rows[1] and f"val/greedy_{key}" in rows[1]
