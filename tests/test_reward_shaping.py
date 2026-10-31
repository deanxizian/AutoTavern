"""Real simulator cycles and PPO returns must include costs without filtering actions."""

import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import taverngym as tg
from stable_baselines3.common.callbacks import BaseCallback
from taverngym.learning.env import TavernGymEnv
from taverngym.learning.model import ModelConfig
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.rewards import ActionRewardShaper, REWARD_METRICS, RewardShapingConfig
from taverngym.learning.vector_env import NativeVectorEnv


COSTS = {"operation_cost": 0.0001, "cycle_cost": 0.0002}
ALL_ACTION_COSTS = {**COSTS, "cycle_detection": "all_actions"}
CYCLE_ONLY_COSTS = {**ALL_ACTION_COSTS, "operation_cost": 0}


def observation(game):
    f, c = game.observe(0)
    return {"features": np.frombuffer(f, dtype="<f4").copy(),
            "cards": np.frombuffer(c, dtype="<i4").copy(),
            "action_mask": np.frombuffer(game.legal_mask(0), dtype=np.uint8).astype(bool)}


@pytest.fixture
def game():
    g = tg.Game(7, json.dumps({"hero_pool": ["TB_BaconShop_HERO_34"], "hero_choices": 1}))
    for _ in range(100):
        if g.turn >= 1:
            break
        for p in range(8):
            if g.needs_action(p):
                g.step(p, g.bot_action(p))
    cards = [c["id"] for c in json.loads(tg.catalog_json())
             if c["kind"] == "minion" and c["in_pool"] and c["implemented"]][:3]
    for zone in ("board", "hand", "shop"):
        g.debug(json.dumps({"cmd": "clear_zone", "p": 0, "zone": zone}))
    for card in cards:
        g.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "board", "card": card}))
    g.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "shop", "card": cards[0]}))
    g.debug(json.dumps({"cmd": "set_player", "p": 0, "gold": 10}))
    return g


def execute(game, shaper, action):
    before = observation(game)
    original = {key: value.copy() for key, value in before.items()}
    assert before["action_mask"][action]
    game.step(0, action)
    after = observation(game)
    reward, metrics = shaper.step(action, before, after, 0)
    assert metrics is None
    for key in before:
        np.testing.assert_array_equal(before[key], original[key])
        np.testing.assert_array_equal(after[key], observation(game)[key])
    return reward


@pytest.mark.parametrize("actions", [[197, 203], [197, 205, 210], [2, 2], [197, 2, 203, 2]])
@pytest.mark.parametrize("costs", [COSTS, ALL_ACTION_COSTS, CYCLE_ONLY_COSTS])
def test_native_move_freeze_and_interleaved_cycles_receive_extra_cost(game, actions, costs):
    shaper = ActionRewardShaper(RewardShapingConfig(**costs))
    rewards = [execute(game, shaper, action) for action in actions]
    assert rewards[:-1] == pytest.approx([-costs["operation_cost"]] * (len(actions) - 1))
    assert rewards[-1] == pytest.approx(-costs["operation_cost"] - costs["cycle_cost"])
    # The reverse action remains a native legal action, even after being penalized.
    assert observation(game)["action_mask"][actions[0]]


def test_productive_action_and_visible_progress_reset_history(game):
    shaper = ActionRewardShaper(RewardShapingConfig(**COSTS))
    assert execute(game, shaper, 197) == pytest.approx(-0.0001)
    assert execute(game, shaper, 1) == 0  # Refresh starts a new segment.
    assert execute(game, shaper, 203) == pytest.approx(-0.0001)
    assert execute(game, shaper, 197) == pytest.approx(-0.0003)
    # Also reset when public resources change between observations without a
    # recorded productive action (e.g. a caller restoring / updating a state).
    game.debug(json.dumps({"cmd": "set_player", "p": 0, "gold": 20}))
    assert execute(game, shaper, 203) == pytest.approx(-0.0001)
    # A progress change caused by the action itself must not count as a cycle.
    before = observation(game)
    game.step(0, 197)
    game.debug(json.dumps({"cmd": "set_player", "p": 0, "gold": 21}))
    after = observation(game)
    assert shaper.step(197, before, after, 0)[0] == pytest.approx(-0.0001)


@pytest.mark.parametrize("costs", [COSTS, ALL_ACTION_COSTS])
def test_completed_episode_and_explicit_reset_do_not_leak_history(game, costs):
    shaper = ActionRewardShaper(RewardShapingConfig(**costs))
    execute(game, shaper, 197)
    execute(game, shaper, 203)
    obs = observation(game)
    reward, totals = shaper.step(0, obs, obs, 1.0, done=True)
    assert reward == 1
    assert totals == pytest.approx(dict(placement_reward=1, operation_penalty=-0.0002,
                                      cycle_penalty=-0.0002, total_reward=0.9996,
                                      reversible_actions=2, cycle_returns=1, actions_per_game=3))
    assert execute(game, shaper, 197) == pytest.approx(-0.0001)
    shaper.reset()
    assert execute(game, shaper, 203) == pytest.approx(-0.0001)


@pytest.mark.parametrize("card,actions", [
    ("BG28_897", [46, 7]),  # Play targeted Banana, then cancel.
    ("BG31_886", [46, 30, 7]),  # Forest's Bounty: play, choose mode, cancel targeting.
])
@pytest.mark.parametrize("costs", [ALL_ACTION_COSTS, CYCLE_ONLY_COSTS])
def test_native_play_choose_cancel_cycles_and_legacy_compatibility(game, card, actions, costs):
    game.debug(json.dumps({"cmd": "add_spell", "p": 0, "zone": "hand", "card": card}))
    shaper = ActionRewardShaper(RewardShapingConfig(**costs))
    original = observation(game)
    rewards = [execute(game, shaper, action) for action in actions * 2]
    # First return closes the loop; subsequently re-entering its known menu
    # states counts too. No flat operation cost is added to these actions.
    assert rewards[:len(actions) - 1] == [0] * (len(actions) - 1)
    assert rewards[len(actions) - 1:] == pytest.approx([-0.0002] * (len(actions) + 1))
    for key in ("features", "cards"):
        np.testing.assert_array_equal(original[key], observation(game)[key])
    assert shaper._totals["reversible_actions"] == 0
    assert shaper._totals["operation_penalty"] == 0
    assert shaper._totals["cycle_returns"] == len(actions) + 1
    legacy = ActionRewardShaper(RewardShapingConfig(**COSTS))
    assert legacy.config.cycle_detection == "reversible_segment"
    assert [execute(game, legacy, action) for action in actions * 2] == [0] * (2 * len(actions))


def test_native_hero_power_target_cancel_cycle_without_operation_cost(game):
    game.debug(json.dumps({"cmd": "set_hero_power", "p": 0, "slot": 0,
                           "card": "TB_BaconShop_HP_011"}))
    shaper = ActionRewardShaper(RewardShapingConfig(**CYCLE_ONLY_COSTS))
    # Galakrond's Greed: open Tavern-minion targeting, then back out.
    assert execute(game, shaper, 4) == 0
    assert execute(game, shaper, 7) == pytest.approx(-0.0002)
    assert execute(game, shaper, 4) == pytest.approx(-0.0002)
    assert shaper._totals["operation_penalty"] == 0


def test_native_successful_spell_and_choice_are_not_repetition(game):
    game.debug(json.dumps({"cmd": "add_spell", "p": 0, "zone": "hand", "card": "BG28_897"}))
    shaper = ActionRewardShaper(RewardShapingConfig(**ALL_ACTION_COSTS))
    assert execute(game, shaper, 46) == 0
    target = int(np.flatnonzero(observation(game)["action_mask"][30:46])[0]) + 30
    before = observation(game)
    assert execute(game, shaper, target) == 0
    assert not np.array_equal(before["features"], observation(game)["features"])
    assert shaper._totals["cycle_returns"] == 0
    assert execute(game, shaper, 197) == pytest.approx(-0.0001)
    assert execute(game, shaper, 203) == pytest.approx(-0.0003)


def test_all_action_history_ignores_mask_but_resets_at_turn_end_and_discontinuities(game):
    shaper = ActionRewardShaper(RewardShapingConfig(**ALL_ACTION_COSTS))
    base = observation(game)
    pending = {k: v.copy() for k, v in base.items()}
    pending["cards"][0] += 1
    restored = {k: v.copy() for k, v in base.items()}
    restored["action_mask"][:] = False
    restored["action_mask"][0] = True
    assert shaper.step(46, base, pending, 0)[0] == 0
    assert shaper.step(7, pending, restored, 0)[0] == pytest.approx(-0.0002)
    assert shaper.step(0, restored, restored, 0)[0] == 0
    assert shaper.step(46, base, pending, 0)[0] == 0
    shaper.reset()
    assert shaper.step(7, pending, base, 0)[0] == 0
    # Same public state on a different turn cannot revisit the prior turn.
    for obs in (base, pending):
        obs["features"][0] += 0.05
    assert shaper.step(46, base, pending, 0)[0] == 0
    # External resource changes must not inherit stale pre-restore history.
    base["features"][7] += 1
    assert shaper.step(7, base, pending, 0)[0] == 0


@pytest.mark.parametrize("costs", [ALL_ACTION_COSTS, CYCLE_ONLY_COSTS])
def test_native_batch_exposes_new_cycle_cost_without_changing_legality(game, costs):
    game.debug(json.dumps({"cmd": "add_spell", "p": 0, "zone": "hand", "card": "BG31_886"}))
    env = NativeVectorEnv(1, seed=7, reward_shaping=costs)
    try:
        env.reset()
        state = json.loads(env.batch.save(0))
        state["game"] = json.loads(game.state_json())
        env.batch.load(0, json.dumps(state))
        env._observe()
        for action, expected in [(46, 0), (30, 0), (7, -0.0002), (46, -0.0002)]:
            assert env.action_masks()[0, action]
            _, rewards, dones, _ = env.step([action])
            assert not dones[0]
            assert float(rewards[0]) == pytest.approx(expected)
        assert env._reward_shapers[0]._totals["cycle_returns"] == 2
    finally:
        env.close()


def test_invalid_cycle_detection_fails_before_sampling():
    with pytest.raises(ValueError, match="cycle_detection"):
        NativeVectorEnv(1, reward_shaping={"cycle_detection": "unknown"})


@pytest.mark.parametrize("costs", [{"operation_cost": -1}, {"cycle_cost": -1},
                                   {"operation_cost": float("nan")}, {"cycle_cost": float("inf")}])
def test_invalid_reward_scales_fail_before_sampling(costs):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        NativeVectorEnv(1, reward_shaping=costs)


def test_native_batch_costs_do_not_change_games_masks_or_placement_and_reset_per_seat():
    raw = NativeVectorEnv(3, seed=66, threads=1)
    shaped = NativeVectorEnv(3, seed=66, threads=3, reward_shaping=COSTS)
    left, right = raw.reset(), shaped.reset()
    completed = 0
    try:
        for _ in range(300):
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])
            actions = [2 if mask[2] else int(np.flatnonzero(mask)[0]) for mask in left["action_mask"]]
            left, native_reward, dn1, info1 = raw.step(actions)
            right, reward, dn2, info2 = shaped.step(actions)
            np.testing.assert_array_equal(dn1, dn2)
            for i, action in enumerate(actions):
                if action == 2:
                    assert native_reward[i] - reward[i] == pytest.approx(0.0001, abs=1e-8) or (
                        native_reward[i] - reward[i] == pytest.approx(0.0003, abs=1e-8))
                else:
                    assert native_reward[i] == reward[i]
                if dn1[i]:
                    a, b = info1[i], info2[i]
                    assert a["placement"] == b["placement"]
                    totals = b["reward_components"]
                    assert totals["placement_reward"] == pytest.approx((9 - 2 * b["placement"]) / 7)
                    assert totals["cycle_returns"] > 0
                    assert totals["operation_penalty"] == pytest.approx(-0.0001 * totals["reversible_actions"])
                    assert totals["cycle_penalty"] == pytest.approx(-0.0002 * totals["cycle_returns"])
                    assert totals["actions_per_game"] == b["episode"]["l"]
                    assert b["episode"]["r"] == pytest.approx(totals["total_reward"], abs=1e-7)
                    assert a["episode"]["l"] == b["episode"]["l"]
                    completed += 1
        assert completed >= 6
    finally:
        raw.close()
        shaped.close()


def test_gym_adapter_emits_costs_and_episode_components():
    raw, shaped = TavernGymEnv(seed=66), TavernGymEnv(seed=66, reward_shaping=COSTS)
    left, _ = raw.reset(seed=66)
    right, _ = shaped.reset(seed=66)
    for _ in range(300):
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        action = 2 if left["action_mask"][2] else int(np.flatnonzero(left["action_mask"])[0])
        left, native_reward, done, _, a = raw.step(action)
        right, reward, other_done, _, b = shaped.step(action)
        assert done == other_done and a["placement"] == b["placement"]
        assert reward <= native_reward
        if done:
            assert b["reward_components"]["cycle_returns"] > 0
            assert b["reward_components"]["placement_reward"] == native_reward
            break
    else:
        pytest.fail("Native game did not finish")


@pytest.mark.parametrize("costs", [COSTS, ALL_ACTION_COSTS])
def test_shaped_rewards_reach_ppo_buffer_gae_and_episode_logs(costs):
    torch.set_num_threads(2)
    env = NativeVectorEnv(1, seed=4, threads=1, config={"turn_action_caps": [3]}, reward_shaping=costs)
    model = EntityPPO(EntityPolicy, env, n_steps=128, batch_size=64, n_epochs=1,
                      gamma=1, gae_lambda=1, seed=4, device="cpu",
                      policy_kwargs={"model_config": asdict(ModelConfig(
                          d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})
    text = torch.randn(tg.N_CARDS, 16)
    text[0].zero_()
    model.policy.network.set_text_embeddings(text)

    class Capture(BaseCallback):
        def __init__(self):
            super().__init__()
            self.records = []

        def _on_step(self):
            self.records.append((self.locals["rewards"].copy(), self.locals["dones"].copy(),
                                 self.locals["infos"]))
            return True

    capture = Capture()
    try:
        _, callback = model._setup_learn(128, capture)
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 128)
        buffer = model.rollout_buffer
        np.testing.assert_array_equal(buffer.rewards, np.stack([row[0] for row in capture.records]))
        actions = buffer.actions[:, 0, 0]
        reversible = (actions == 2) | (actions >= dict(tg.layout())["A_MOVE"])
        assert reversible.any()
        native_rewards = np.array([
            (9 - 2 * row[2][0]["placement"]) / 7 if row[1][0] else 0 for row in capture.records
        ])
        assert (native_rewards - buffer.rewards[:, 0])[reversible].min() >= 0.000099
        if costs == COSTS:
            np.testing.assert_allclose(native_rewards[~reversible], buffer.rewards[~reversible, 0], atol=1e-7)
        else:
            assert np.all(buffer.rewards[:, 0] <= native_rewards + 1e-7)
            assert np.all(native_rewards - buffer.rewards[:, 0] <= 0.000301)
        with torch.no_grad():
            tensors, _ = model.policy.obs_to_tensor(model._last_obs)
            last_value = float(model.policy.predict_values(tensors).item())
        expected, continuation = np.zeros(128), last_value
        for i in reversed(range(128)):
            continuation = float(buffer.rewards[i, 0]) + (0 if capture.records[i][1][0] else continuation)
            expected[i] = continuation
        np.testing.assert_allclose(buffer.returns[:, 0], expected, atol=2e-6)
        np.testing.assert_allclose(buffer.advantages[:, 0], expected - buffer.values[:, 0], atol=2e-6)
        assert model._rollout_reward_components
        assert all(f"rollout/{name}" in model.logger.name_to_value for name in REWARD_METRICS)
    finally:
        env.close()
