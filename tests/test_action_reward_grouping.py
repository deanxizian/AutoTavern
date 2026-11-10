"""Compare against the actual archived shaper: grouping must not change rewards."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from stable_baselines3.common.callbacks import BaseCallback
from taverngym.learning import rewards
from taverngym.learning.actor_sampling import ActorOutput, UnifiedActorSampler
from taverngym.learning.experiment import log_evaluation
from taverngym.learning.model import ModelConfig, ObservationLayout
from taverngym.learning.opponents import FrozenOpponentPool, OpponentGroup
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.stage_chain import digest
from taverngym.learning.tracking import WandbTracker
from taverngym.learning.vector_env import NativeVectorEnv


ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / ("experiments/20261031-infra/20261031-training-agents/instances/"
                   "20261101-v789-move0005-256x512-01")
BASELINE = json.loads((INSTANCE / "action-reward-baseline.json").read_text())
BASELINE_PATH = ROOT / BASELINE["path"]
assert digest(BASELINE_PATH) == BASELINE["sha256"]
SPEC = importlib.util.spec_from_file_location(
    "taverngym.learning._pre_action_grouping", BASELINE_PATH)
OLD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OLD
SPEC.loader.exec_module(OLD)
CONFIG_PATH = BASELINE_PATH.parents[3] / (
    "experiments/20261101-exp07-v7-history-ranked/"
    "20261101-ppo-v7-move0005-cycle001-forceend-256x512/config.json")
assert digest(CONFIG_PATH) == BASELINE["prepared_config_sha256"]["v7"]
CONFIG = json.loads(CONFIG_PATH.read_text())
PROTOCOL = CONFIG["reward_shaping"]
LAYOUT = ObservationLayout()
NATIVE = LAYOUT.native
PENDING = NATIVE["GLOBAL_F"] + NATIVE["N_QUESTS"] * NATIVE["QUEST_F"]
HEROES = ["TB_BaconShop_HERO_10", "TB_BaconShop_HERO_37", "TB_BaconShop_HERO_34"]
MEANS = {rewards._active_hero_indices()[name]: value
         for name, value in PROTOCOL["hero_selection_mean_placements"].items()}
SIMULATOR = {"hero_pool": HEROES, "hero_choices": 3, "turn_action_caps": [60]}


@pytest.fixture(autouse=True)
def no_adam(monkeypatch):
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *a, **k: pytest.fail("No Adam allowed"))


def identity(metrics):
    assert metrics["action_reward"] == pytest.approx(
        sum(metrics.get(name, 0.) for name in rewards.ACTION_REWARD_COMPONENTS), abs=1e-12)
    assert metrics["total_reward"] == pytest.approx(
        metrics["placement_reward"] + metrics["hero_selection_bonus"] + metrics["action_reward"],
        abs=1e-12)


def equivalent(old, new):
    assert {key: value for key, value in new.items() if key != "action_reward"} == old
    identity(new)


def actions(obs):
    chosen = obs["action_mask"].argmax(axis=1)
    for lane, features in enumerate(obs["features"]):
        mask = obs["action_mask"][lane]
        if features[PENDING + 1]:
            offers = obs["cards"][lane, LAYOUT.cards["choices"]]
            options = np.flatnonzero(mask[NATIVE["A_CHOOSE"]:NATIVE["A_PLAY"]])
            best = min(options, key=lambda i: MEANS[int(offers[i])])
            chosen[lane] = NATIVE["A_CHOOSE"] + best
        elif features[PENDING + 2]:
            chosen[lane] = NATIVE["A_CHOOSE"]
        elif round(float(features[0]) * 20) == 1:
            # A real buy/play milestone, then native forced End after looping.
            plays = np.flatnonzero(mask[NATIVE["A_PLAY"]:NATIVE["A_MAGNETIZE"]])
            buys = np.flatnonzero(mask[NATIVE["A_BUY"]:NATIVE["A_SELL"]])
            if len(plays):
                chosen[lane] = NATIVE["A_PLAY"] + plays[0]
            elif not features[33] and len(buys):
                span, count, width = LAYOUT.floats["shop"]
                shop = features[span].reshape(count, width)
                minions = [i for i in buys if shop[i, 1]]
                chosen[lane] = NATIVE["A_BUY"] + minions[0] if minions else 2 if mask[2] else 0
            elif mask[2]:
                chosen[lane] = 2
        elif round(float(features[0]) * 20) == 2 and mask[NATIVE["A_UPGRADE"]]:
            chosen[lane] = NATIVE["A_UPGRADE"]
        elif round(float(features[0]) * 20) == 3 and mask[NATIVE["A_DARK_GIFT"]]:
            chosen[lane] = NATIVE["A_DARK_GIFT"]
    return chosen


def replace_with_old(env, protocol):
    env._reward_shapers = [OLD.ActionRewardShaper(OLD.RewardShapingConfig(**protocol))
                           for _ in range(env.num_envs)]


@pytest.mark.parametrize("protocol", [PROTOCOL,
    {"operation_cost": .0001, "cycle_cost": .0002},
    {**PROTOCOL, "cycle_forfeit_repetitions": 3},
    {**PROTOCOL, "turn_action_forfeit_limit": 4}])
def test_pre_refactor_native_replay_preserves_reward_components_state_and_placement(protocol):
    left, right = [NativeVectorEnv(1, seed=13, threads=1, config=SIMULATOR,
                  reward_shaping=protocol) for _ in range(2)]
    a, b = left.reset(), right.reset()
    replace_with_old(left, protocol)
    completed = 0
    try:
        for _ in range(160):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            selected = actions(a)
            a, ar, ad, ai = left.step(selected)
            b, br, bd, bi = right.step(selected)
            # float32 reward equality is exact, not merely close after regrouping.
            np.testing.assert_array_equal(ar, br)
            np.testing.assert_array_equal(ad, bd)
            assert left.batch.save(0) == right.batch.save(0)
            if ad[0]:
                completed += 1
                assert ai[0]["placement"] == bi[0]["placement"]
                assert ai[0]["episode"]["r"] == bi[0]["episode"]["r"]
                equivalent(ai[0]["reward_components"], bi[0]["reward_components"])
                break
            equivalent(left._reward_shapers[0]._totals, right._reward_shapers[0]._totals)
        assert completed == 1
    finally:
        left.close()
        right.close()


def test_unified_actor_replay_preserves_old_transitions_across_two_collects():
    class Actor:
        def act(self, obs):
            selected = actions(obs)
            return ActorOutput(selected, np.zeros(len(selected), np.float32),
                               np.zeros(len(selected), np.float32))

    def sampler():
        opponents = FrozenOpponentPool((OpponentGroup("fixed", Actor(), 1, PROTOCOL),))
        return UnifiedActorSampler(1, Actor(), opponents, frozen_opponents=1, seed=13,
                                   threads=1, config=SIMULATOR, reward_shaping=PROTOCOL)

    left, right = sampler(), sampler()
    left.shapers = [OLD.ActionRewardShaper(OLD.RewardShapingConfig(**PROTOCOL))]
    left.opponent_shapers = {slot: OLD.ActionRewardShaper(OLD.RewardShapingConfig(**PROTOCOL))
                            for slot in left.opponent_shapers}
    a, b = [], []
    for _ in range(2):
        left.collect(80, a.append)
        right.collect(80, b.append)
        assert left.batch.save(0) == right.batch.save(0)
    for old, new in zip(a, b):
        assert old.action == new.action and old.reward == new.reward
        assert old.done == new.done and old.placement == new.placement
        if old.done:
            equivalent(old.info["reward_components"], new.info["reward_components"])
    assert len(a) == len(b) == 160


def test_ppo_reward_and_gae_arrays_are_bit_identical_to_actual_baseline(monkeypatch):
    torch.set_num_threads(2)
    results = []

    class Continue(BaseCallback):
        def _on_step(self):
            return True

    for old in (True, False):
        env = NativeVectorEnv(1, seed=13, threads=1, config=SIMULATOR, reward_shaping=PROTOCOL)
        model = EntityPPO(EntityPolicy, env, n_steps=96, batch_size=32, n_epochs=1,
            gamma=.99, gae_lambda=.95, device="cpu", seed=13, policy_kwargs={"model_config": asdict(
                ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})

        def forward(obs, deterministic=False, action_masks=None):
            arrays = {key: value.detach().cpu().numpy() for key, value in obs.items()}
            selected = torch.from_numpy(actions(arrays))
            return selected, torch.zeros((1, 1)), torch.zeros(1)

        monkeypatch.setattr(model.policy, "forward", forward)
        monkeypatch.setattr(model.policy, "predict_values", lambda obs: torch.zeros((1, 1)))
        try:
            _, callback = model._setup_learn(96, Continue())
            if old:
                replace_with_old(env, PROTOCOL)
            assert model.collect_rollouts(env, callback, model.rollout_buffer, 96)
            results.append([getattr(model.rollout_buffer, key).copy()
                            for key in ("rewards", "returns", "advantages", "actions")])
            assert model.optimizer_step_count() == 0 and not model.policy.optimizer.state
        finally:
            env.close()
    for old, new in zip(*results):
        np.testing.assert_array_equal(old, new)


def test_derived_action_metric_is_logged_once_and_all_details_are_retained(tmp_path, monkeypatch):
    recorded = []
    run = SimpleNamespace(entity="eric_eric_", project="AutoTavern", id="mock", url="mock",
        define_metric=lambda *a, **k: None, log=recorded.append, finish=lambda **k: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **k: run,
                                                             Settings=lambda **k: k))
    logger_values = {}
    row = dict.fromkeys(rewards.REWARD_METRICS, 0.)
    row.update(mean_placement=3., first_rate=.2, top4_rate=.7, action_reward=-.01)
    tracker = WandbTracker(tmp_path, {"tracking": {"mode": "offline"}})
    try:
        tracker.log({f"rollout/{key}": value for key, value in row.items()}, 0)
        log_evaluation(SimpleNamespace(record=lambda k, v: logger_values.update({k: v})), "val",
                       {"conditions": {"mixed_sampled": row, "mixed_greedy": row}})
        tracker.log(logger_values, 0)
    finally:
        tracker.finish()
    for key in ("action_reward", *rewards.ACTION_REWARD_COMPONENTS, "hero_selection_bonus"):
        assert f"train/{key}" in recorded[0]
        assert f"val/{key}" in recorded[1] and f"val/greedy_{key}" in recorded[1]


@pytest.mark.parametrize("finish", ["terminal", "forfeit"])
def test_finish_adds_only_native_reward_without_reapplying_action_subtotal(finish):
    # A valid actual native observation exercises existing cycle bookkeeping.
    env = NativeVectorEnv(1, seed=13, config=SIMULATOR, reward_shaping=PROTOCOL)
    state = env.reset()
    old = OLD.ActionRewardShaper(OLD.RewardShapingConfig(**PROTOCOL))
    new = rewards.ActionRewardShaper(rewards.RewardShapingConfig(**PROTOCOL))
    try:
        for _ in range(16):
            before = {key: value[0].copy() for key, value in state.items()}
            selected = actions(state)
            state, _, done, _ = env.step(selected)
            assert not done[0]
            after = {key: value[0].copy() for key, value in state.items()}
            assert old.step(int(selected[0]), before, after, 0)[0] == new.step(
                int(selected[0]), before, after, 0)[0]
        if finish == "forfeit":
            # Upgrade/End can clear the cycle turn. An actual current-turn
            # reversible action establishes the forfeit bookkeeping precondition.
            before = {key: value[0].copy() for key, value in state.items()}
            assert before["action_mask"][2]
            state, _, done, _ = env.step([2])
            assert not done[0]
            after = {key: value[0].copy() for key, value in state.items()}
            assert old.step(2, before, after, 0)[0] == new.step(2, before, after, 0)[0]
        subtotal = new._totals["action_reward"]
        assert new._totals["early_action_penalty"] <= -.01
        assert new._totals["cycle_penalty"] < -.01  # No total action cap.
        if finish == "forfeit":
            old.forfeit_requested = new.forfeit_requested = True
            a, b = old.finish_forfeit(-1, 8), new.finish_forfeit(-1, 8)
        else:
            a, b = old.finish_terminal(-1, 8), new.finish_terminal(-1, 8)
        equivalent(a, b)
        assert b["action_reward"] == subtotal
        assert b["opening_bonus"] > 0
        assert b["hero_selection_bonus"] > 0 and b["hero_selection_bonus"] not in (
            b["action_reward"], b["opening_bonus"])
    finally:
        env.close()
