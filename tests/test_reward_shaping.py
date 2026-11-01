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
FORFEIT_COSTS = {**CYCLE_ONLY_COSTS, "cycle_forfeit_repetitions": 3}
SOFT_CYCLE_COSTS = {"operation_cost": 0, "cycle_cost": .01,
                    "cycle_detection": "all_actions", "cycle_forfeit_repetitions": None,
                    "turn_action_forfeit_limit": 60}


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
                                      reversible_actions=2, cycle_returns=1, actions_per_game=3,
                                      cycle_forfeit_rate=0, cycle_forfeit_turn=0,
                                      cycle_forfeit_placement=0, max_consecutive_cycles=0,
                                      action_limit_forfeit_rate=0, action_limit_forfeit_turn=0,
                                      action_limit_forfeit_placement=0,
                                      opponent_cycle_forfeits=0, opponent_action_limit_forfeits=0,
                                      cycle_return_fraction=1/3, max_consecutive_cycle_returns=1,
                                      move_actions=2, move_penalty=0, move_action_fraction=2/3))
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


@pytest.mark.parametrize("actions,spell,power", [
    ([2, 2], None, None),
    ([197, 203], None, None),
    ([46, 7], "BG28_897", None),
    ([46, 30, 7], "BG31_886", None),
    ([4, 7], None, "TB_BaconShop_HP_011"),
])
def test_cycle_forfeit_counts_three_complete_round_trips(game, actions, spell, power):
    if spell:
        game.debug(json.dumps({"cmd": "add_spell", "p": 0, "zone": "hand", "card": spell}))
    if power:
        game.debug(json.dumps({"cmd": "set_hero_power", "p": 0, "slot": 0, "card": power}))
    shaper = ActionRewardShaper(RewardShapingConfig(**FORFEIT_COSTS))
    for index, action in enumerate(actions * 3):
        execute(game, shaper, action)
        assert shaper.consecutive_cycles == (index + 1) // len(actions)
        assert shaper.forfeit_requested == (index + 1 == len(actions) * 3)
    # The cost counts every return; the forfeit counts complete round trips.
    assert shaper._totals["cycle_returns"] == 2 * len(actions) + 1


@pytest.mark.parametrize("boundary", ["end", "reset", "turn", "discontinuity", "progress"])
def test_cycle_forfeit_streak_resets_on_progress_and_boundaries(game, boundary):
    shaper = ActionRewardShaper(RewardShapingConfig(**FORFEIT_COSTS))
    for action in [2, 2] * 2:
        execute(game, shaper, action)
    assert shaper.consecutive_cycles == 2
    base = observation(game)
    if boundary == "reset":
        shaper.reset()
    elif boundary == "end":
        shaper.step(0, base, base, 0)
    elif boundary == "turn":
        changed = {key: value.copy() for key, value in base.items()}
        changed["features"][0] += 0.05
        shaper.step(2, base, changed, 0)
    elif boundary == "discontinuity":
        game.debug(json.dumps({"cmd": "set_player", "p": 0, "gold": 20}))
        execute(game, shaper, 2)
    else:
        before = observation(game)
        game.step(0, 2)
        game.debug(json.dumps({"cmd": "set_player", "p": 0, "gold": 20}))
        shaper.step(2, before, observation(game), 0)
    assert shaper.consecutive_cycles == 0
    assert not shaper.forfeit_requested


def test_native_vector_cycle_forfeit_is_terminal_and_charges_trigger_action(game):
    env = NativeVectorEnv(2, seed=7, reward_shaping=FORFEIT_COSTS)
    try:
        env.reset()
        state = json.loads(env.batch.save(0))
        state["game"] = json.loads(game.state_json())
        env.batch.load(0, json.dumps(state))
        env._observe()
        for index in range(6):
            other = int(np.flatnonzero(env.action_masks()[1])[0])
            obs, rewards, dones, infos = env.step([2, other])
            assert bool(dones[0]) == (index == 5)
        info = infos[0]
        assert info["termination_reason"] == "cycle_forfeit"
        assert info["placement"] == 8
        assert info["TimeLimit.truncated"] is False
        assert rewards[0] == pytest.approx(-1.0002)
        components = info["reward_components"]
        assert components["cycle_forfeit_rate"] == 1
        assert components["cycle_forfeit_turn"] == 1
        assert components["cycle_forfeit_placement"] == 8
        assert components["max_consecutive_cycles"] == 3
        assert components["cycle_returns"] == 5
        assert components["placement_reward"] == -1
        assert components["operation_penalty"] == 0
        assert components["total_reward"] == pytest.approx(-1.001)
        assert info["episode"]["l"] == 6
        assert info["episode"]["r"] == pytest.approx(-1.001)
        assert env.batch.episode_index(0) == 1
        assert obs["action_mask"][0].any()
        assert not env._reward_shapers[0].forfeit_requested
    finally:
        env.close()


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "3"])
def test_invalid_cycle_forfeit_count_fails_before_sampling(count):
    with pytest.raises(ValueError, match="positive integer"):
        RewardShapingConfig(**{**FORFEIT_COSTS, "cycle_forfeit_repetitions": count})


@pytest.mark.parametrize("costs", [COSTS, ALL_ACTION_COSTS, CYCLE_ONLY_COSTS])
def test_legacy_reward_config_never_enables_forfeits(game, costs):
    shaper = ActionRewardShaper(RewardShapingConfig(**costs))
    for _ in range(8):
        execute(game, shaper, 2)
    assert not shaper.forfeit_requested
    with pytest.raises(ValueError, match="all_actions"):
        RewardShapingConfig(cycle_forfeit_repetitions=3)


def test_gym_cycle_forfeit_finishes_without_time_limit_truncation():
    env = TavernGymEnv(seed=7, config={"turn_action_caps": [60]}, reward_shaping=FORFEIT_COSTS)
    try:
        obs, _ = env.reset()
        for _ in range(100):
            action = 2 if obs["action_mask"][2] else int(np.flatnonzero(obs["action_mask"])[0])
            obs, reward, done, truncated, info = env.step(action)
            if done:
                assert not truncated
                assert info["termination_reason"] == "cycle_forfeit"
                assert info["placement"] == 8
                assert reward == pytest.approx(-1.0002)
                assert info["reward_components"]["max_consecutive_cycles"] == 3
                break
        else:
            pytest.fail("Three Freeze/unfreeze round trips did not trigger surrender")
    finally:
        env.close()


def test_sixty_action_budget_forfeits_attempt_61_instead_of_forced_end():
    costs = {**CYCLE_ONLY_COSTS, "turn_action_forfeit_limit": 60}
    config = {"turn_action_caps": [60], "freeze_presses_per_turn": 0}
    env = NativeVectorEnv(1, seed=7, config=config, reward_shaping=costs)
    try:
        obs = env.reset()
        while obs["features"][0, 0] == 0:
            obs, _, dones, _ = env.step([int(np.flatnonzero(obs["action_mask"][0])[0])])
            assert not dones[0]
        for _ in range(60):
            obs, _, dones, _ = env.step([2])
            assert not dones[0]
        assert obs["action_mask"][0].sum() == 1 and obs["action_mask"][0, 0]
        # The forced End must be intercepted before combat can improve placement.
        _, rewards, dones, infos = env.step([0])
        assert dones[0]
        assert rewards[0] == -1
        info = infos[0]
        assert info["termination_reason"] == "action_limit_forfeit"
        assert info["placement"] == 8
        assert info["TimeLimit.truncated"] is False
        assert info["reward_components"]["cycle_forfeit_rate"] == 0
        assert info["reward_components"]["action_limit_forfeit_rate"] == 1
        assert info["reward_components"]["action_limit_forfeit_turn"] == 1
        assert info["reward_components"]["action_limit_forfeit_placement"] == 8
        assert info["reward_components"]["cap_end_rate"] == 0
    finally:
        env.close()


def test_fixed_cent_per_return_keeps_loops_alive_and_records_per_game_repeat_metrics():
    env = NativeVectorEnv(1, seed=7, reward_shaping=SOFT_CYCLE_COSTS,
        config={"turn_action_caps": [60], "freeze_presses_per_turn": 0,
                "hero_pool": ["TB_BaconShop_HERO_34"], "hero_choices": 1})
    try:
        obs = env.reset()
        while obs["features"][0, 0] == 0:
            obs, _, dones, _ = env.step([int(np.flatnonzero(obs["action_mask"][0])[0])])
            assert not dones[0]
        penalties = []
        for index in range(60):
            obs, rewards, dones, _ = env.step([2])
            penalties.append(float(rewards[0]))
            assert not dones[0] and not env._reward_shapers[0].forfeit_requested
        assert penalties[0] == 0
        assert penalties[1:] == pytest.approx([-.01] * 59)
        assert obs["action_mask"][0].sum() == 1 and obs["action_mask"][0, 0]
        _, reward, dones, infos = env.step([0])
        assert dones[0] and reward[0] == -1
        metrics = infos[0]["reward_components"]
        assert infos[0]["termination_reason"] == "action_limit_forfeit"
        assert metrics["cycle_forfeit_rate"] == 0
        assert metrics["action_limit_forfeit_rate"] == 1
        assert metrics["cycle_returns"] == 59
        assert metrics["cycle_penalty"] == pytest.approx(-.01 * metrics["cycle_returns"])
        assert metrics["max_consecutive_cycle_returns"] == 59
        assert metrics["cycle_return_fraction"] == pytest.approx(59 / metrics["actions_per_game"])
    finally:
        env.close()


def test_repeating_refresh_with_new_visible_state_costs_nothing(game):
    shaper = ActionRewardShaper(RewardShapingConfig(**SOFT_CYCLE_COSTS))
    for _ in range(3):
        assert execute(game, shaper, 1) == 0
        assert shaper.consecutive_returns == 0
    assert shaper._totals["cycle_penalty"] == shaper._totals["cycle_returns"] == 0


def test_new_progress_end_reset_and_turn_clear_consecutive_returns(game):
    shaper = ActionRewardShaper(RewardShapingConfig(**SOFT_CYCLE_COSTS))
    for _ in range(6):
        execute(game, shaper, 2)
    assert shaper.consecutive_returns == 5
    assert execute(game, shaper, 1) == 0
    assert shaper.consecutive_returns == 0
    execute(game, shaper, 2)
    execute(game, shaper, 2)
    assert shaper.consecutive_returns == 1
    before = observation(game)
    after = {key: value.copy() for key, value in before.items()}
    after["features"][0] += .05
    shaper.step(2, before, after, 0)
    assert shaper.consecutive_returns == 0
    shaper.step(0, after, after, 0)
    assert shaper.consecutive_returns == 0
    shaper.reset()
    assert shaper.consecutive_returns == 0


def test_soft_penalties_at_cap_use_native_forced_end_and_continue_combat():
    costs = {**SOFT_CYCLE_COSTS, "turn_action_forfeit_limit": None}
    env = NativeVectorEnv(1, seed=7, reward_shaping=costs,
        config={"turn_action_caps": [60], "freeze_presses_per_turn": 0,
                "hero_pool": ["TB_BaconShop_HERO_34"], "hero_choices": 1})
    try:
        obs = env.reset()
        while obs["features"][0, 0] == 0:
            obs, _, dones, _ = env.step([int(obs["action_mask"][0].argmax())])
            assert not dones[0]
        for _ in range(60):
            obs, _, dones, _ = env.step([2])
            assert not dones[0]
        assert obs["action_mask"][0].sum() == 1 and obs["action_mask"][0, 0]
        obs, _, dones, infos = env.step([0])
        assert not dones[0] and not infos[0].get("termination_reason")
        assert round(float(obs["features"][0, 0]) * 20) == 2
        assert env.batch.episode_index(0) == 0
        assert env._reward_shapers[0].consecutive_returns == 0
        assert env._reward_shapers[0]._totals["cycle_penalty"] == pytest.approx(-.59)
    finally:
        env.close()


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_turn_action_forfeit_limit(limit):
    with pytest.raises(ValueError, match="turn_action_forfeit_limit"):
        RewardShapingConfig(turn_action_forfeit_limit=limit)


@pytest.mark.parametrize("reason,costs", [
    ("cycle_forfeit", FORFEIT_COSTS),
    ("action_limit_forfeit", {**CYCLE_ONLY_COSTS, "turn_action_forfeit_limit": 4}),
])
def test_forfeit_ppo_returns_do_not_bootstrap_into_reset_game(monkeypatch, reason, costs):
    torch.set_num_threads(2)
    env = NativeVectorEnv(1, seed=7, config={"turn_action_caps": [60]}, reward_shaping=costs)
    model = EntityPPO(EntityPolicy, env, n_steps=32, batch_size=32, n_epochs=1,
                      gamma=1, gae_lambda=1, seed=7, device="cpu",
                      policy_kwargs={"model_config": asdict(ModelConfig(
                          d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})

    def freeze_policy(obs, deterministic=False, action_masks=None):
        masks = obs["action_mask"].bool()
        actions = masks.long().argmax(dim=1)
        actions[masks[:, 2]] = 2
        return actions, torch.full((len(actions), 1), 123.0), torch.zeros(len(actions))

    monkeypatch.setattr(model.policy, "forward", freeze_policy)
    monkeypatch.setattr(model.policy, "predict_values", lambda obs: torch.full((1, 1), 123.0))

    class Capture(BaseCallback):
        def __init__(self):
            super().__init__()
            self.records = []

        def _on_step(self):
            self.records.append((bool(self.locals["dones"][0]), self.locals["infos"][0]))
            return True

    capture = Capture()
    try:
        _, callback = model._setup_learn(32, capture)
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 32)
        buffer = model.rollout_buffer
        terminals = [i for i, (done, _) in enumerate(capture.records) if done]
        assert len(terminals) >= 3
        for i in terminals:
            assert capture.records[i][1]["termination_reason"] == reason
            assert buffer.returns[i, 0] == pytest.approx(buffer.rewards[i, 0], abs=1e-5)
            assert buffer.placement_targets[i, 0] == 7
        expected, continuation = np.zeros(32), 123.0
        for i in reversed(range(32)):
            continuation = float(buffer.rewards[i, 0]) + (
                0 if capture.records[i][0] else continuation)
            expected[i] = continuation
        np.testing.assert_allclose(buffer.returns[:, 0], expected, atol=2e-5)
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


@pytest.mark.parametrize("cost", [-.0005, float("nan"), float("inf"), -float("inf")])
def test_invalid_move_cost_fails_before_sampling(cost):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        RewardShapingConfig(move_cost=cost)


@pytest.mark.parametrize("move_cost", [0, .0005])
@pytest.mark.parametrize("operation_cost", [0, .0001])
def test_native_move_cost_is_separate_from_freeze_and_adds_to_cycle(
        game, move_cost, operation_cost):
    costs = {**CYCLE_ONLY_COSTS, "cycle_cost": .01, "operation_cost": operation_cost}
    # Omission and explicit zero have identical historical reward behavior.
    if move_cost:
        costs["move_cost"] = move_cost
    shaper = ActionRewardShaper(RewardShapingConfig(**costs))
    assert execute(game, shaper, 197) == pytest.approx(-operation_cost - move_cost)
    assert execute(game, shaper, 203) == pytest.approx(-operation_cost - move_cost - .01)
    assert execute(game, shaper, 2) == pytest.approx(-operation_cost)
    totals = shaper.finish_terminal(1, 1)
    assert totals["move_actions"] == 2
    assert totals["move_penalty"] == pytest.approx(-2 * move_cost)
    assert totals["move_action_fraction"] == pytest.approx(2/3)
    assert totals["operation_penalty"] == pytest.approx(-3 * operation_cost)
    assert totals["cycle_penalty"] == pytest.approx(-.01)
    assert totals["total_reward"] == pytest.approx(1 - 3 * operation_cost - 2 * move_cost - .01)
    assert shaper._totals["move_actions"] == shaper._totals["move_penalty"] == 0
    assert execute(game, shaper, 197) == pytest.approx(-operation_cost - move_cost)
    shaper.reset()
    assert shaper.finish_terminal(0, 8)["move_action_fraction"] == 0


def test_executed_terminal_move_charged_but_intercepted_request_and_turn0_are_not(game):
    costs = {**CYCLE_ONLY_COSTS, "move_cost": .0005, "cycle_cost": .01}
    before = observation(game)
    game.step(0, 197)  # Real legal Move before a terminal/reset observation.
    after = observation(game)
    shaper = ActionRewardShaper(RewardShapingConfig(**costs))
    reward, totals = shaper.step(197, before, after, -1, done=True)
    assert reward == pytest.approx(-1.0005)
    assert totals["move_actions"] == totals["actions_per_game"] == 1
    assert totals["move_penalty"] == -.0005 and totals["move_action_fraction"] == 1
    assert totals["cycle_penalty"] == 0
    reward, totals = shaper.step(197, before, after, -1, done=True,
                                 terminal_reason="action_limit_forfeit")
    assert reward == -1 and totals["move_actions"] == totals["move_penalty"] == 0
    hero = {key: value.copy() for key, value in before.items()}
    hero["features"][0] = 0
    assert shaper.step(197, hero, after, 0)[0] == 0
    assert shaper._totals["move_actions"] == 0


def test_move_cost_preserves_native_cycle_forfeit_trigger_and_final_accounting(game):
    shaper = ActionRewardShaper(RewardShapingConfig(**{
        **CYCLE_ONLY_COSTS, "move_cost": .0005, "cycle_cost": .01,
        "cycle_forfeit_repetitions": 1}))
    assert execute(game, shaper, 197) == pytest.approx(-.0005)
    assert execute(game, shaper, 203) == pytest.approx(-.0105)
    assert shaper.forfeit_requested
    totals = shaper.finish_forfeit(-1, 8)
    assert totals["actions_per_game"] == totals["move_actions"] == 2
    assert totals["cycle_forfeit_rate"] == 1
    assert totals["move_penalty"] == -.001
    assert totals["total_reward"] == pytest.approx(-1.011)


def test_move_penalty_reaches_ppo_rewards_and_gae_without_optimizer_step(game, monkeypatch):
    torch.set_num_threads(2)
    env = NativeVectorEnv(1, seed=7, threads=1, reward_shaping={
        **CYCLE_ONLY_COSTS, "move_cost": .0005, "cycle_cost": .01})
    model = EntityPPO(EntityPolicy, env, n_steps=2, batch_size=2, n_epochs=1,
        gamma=1, gae_lambda=1, seed=7, device="cpu",
        policy_kwargs={"model_config": asdict(ModelConfig(
            d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})

    def move_policy(obs, deterministic=False, action_masks=None):
        masks = obs["action_mask"].bool()
        actions = torch.full((len(masks),), 197, dtype=torch.long)
        assert masks[torch.arange(len(masks)), actions].all()
        return actions, torch.zeros((len(masks), 1)), torch.zeros(len(masks))

    def forbidden_update(*args, **kwargs):
        pytest.fail("Preparation must never execute an optimizer step")

    class ContinueCollection(BaseCallback):
        def _on_step(self):
            return True

    monkeypatch.setattr(model.policy, "forward", move_policy)
    monkeypatch.setattr(model.policy, "predict_values", lambda obs: torch.zeros((1, 1)))
    monkeypatch.setattr(model.policy.optimizer, "step", forbidden_update)
    parameters = {key: value.clone() for key, value in model.policy.state_dict().items()}
    try:
        _, callback = model._setup_learn(2, ContinueCollection())
        state = json.loads(env.batch.save(0))
        state["game"] = json.loads(game.state_json())
        env.batch.load(0, json.dumps(state))
        model._last_obs = env._observe()
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 2)
        np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0], [-.0005, -.0105], atol=1e-8)
        np.testing.assert_allclose(model.rollout_buffer.returns[:, 0], [-.011, -.0105], atol=1e-8)
        assert env._reward_shapers[0]._totals["move_actions"] == 2
        assert model.optimizer_step_count() == 0 and not model.policy.optimizer.state
        for key, value in model.policy.state_dict().items():
            assert torch.equal(parameters[key], value)
    finally:
        env.close()


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
