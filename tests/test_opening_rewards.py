"""Opening action costs and once-only milestones, with no optimizer updates."""

from dataclasses import asdict
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import taverngym as tg
from stable_baselines3.common.callbacks import BaseCallback
from taverngym.learning.actor_sampling import ActorOutput, UnifiedActorSampler
from taverngym.learning.experiment import log_evaluation
from taverngym.learning.model import ModelConfig, ObservationLayout
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.opponents import FrozenOpponentPool, OpponentGroup
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.rewards import (
    ActionRewardShaper, OPENING_BONUS_FIELDS, REWARD_METRICS, RewardShapingConfig,
    reward_protocols_equal,
)
from taverngym.learning.tracking import WandbTracker
from taverngym.learning.vector_env import NativeVectorEnv


OPENING = {"early_action_costs": [.001, .001, .0005, .0005, .0005],
           "early_action_penalty_cap": .01, "first_turn_minion_bonus": .01,
           "second_turn_tier2_bonus": .005, "third_turn_dark_gift_bonus": .005}
COSTS = {"operation_cost": 0, "cycle_cost": .01, "move_cost": .0005,
         "cycle_detection": "all_actions", **OPENING}
LAYOUT = ObservationLayout()
NATIVE = LAYOUT.native


@pytest.fixture(autouse=True)
def no_adam(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Opening preparation must never update Adam")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)


def obs(turn=1, board=0, tier=1, uses=0, pending=0, source=0):
    features = np.zeros(tg.OBS_FLOATS, np.float32)
    features[0], features[1], features[4] = turn / 20, turn == 0, 1
    features[10], features[23], features[33] = tier / 6, uses / 3, board / 7
    span, count, width = LAYOUT.floats["board"]
    features[span].reshape(count, width)[:board, 0] = 1
    offset = NATIVE["GLOBAL_F"] + NATIVE["N_QUESTS"] * NATIVE["QUEST_F"]
    features[offset + pending] = 1
    cards = np.zeros(tg.OBS_CARDS, np.int32)
    cards[1 + NATIVE["N_POWERS"] + 1] = source
    return {"features": features, "cards": cards, "action_mask": np.ones(tg.N_ACTIONS, bool)}


def shaper(**overrides):
    return ActionRewardShaper(RewardShapingConfig(**{
        **COSTS, "cycle_cost": 0, "move_cost": 0, **overrides}))


@pytest.mark.parametrize("turn,rate", [(0, 0), (1, .001), (2, .001), (3, .0005),
                                      (4, .0005), (5, .0005), (6, 0)])
def test_every_successful_non_end_action_costs_from_first_action(turn, rate):
    tracker = shaper()
    before = obs(turn)
    for index in range(3):
        after = obs(turn)
        after["features"][7] = index + 1  # Productive new state, not a cycle.
        assert tracker.step(NATIVE["A_REFRESH"], before, after, 0)[0] == pytest.approx(-rate)
        before = after
    assert tracker._totals["early_action_penalty"] == pytest.approx(-3 * rate)
    assert tracker._totals["early_chargeable_actions"] == (3 if 1 <= turn <= 5 else 0)


def test_end_counts_in_total_but_not_charged_and_turn_boundaries_change_rate():
    tracker = shaper()
    assert tracker.step(2, obs(2), obs(2), 0)[0] == -.001
    assert tracker.step(0, obs(2), obs(3), 0)[0] == 0
    assert tracker.step(NATIVE["A_CHOOSE"], obs(3), obs(3), 0)[0] == -.0005
    assert tracker.step(0, obs(3), obs(4), 0)[0] == 0
    assert tracker.step(2, obs(5), obs(5), 0)[0] == -.0005
    assert tracker.step(0, obs(5), obs(6), 0)[0] == 0
    assert tracker.step(2, obs(6), obs(6), 0)[0] == 0
    totals = tracker.finish_terminal(1, 1)
    assert totals["early_actions"] == 6
    assert totals["early_chargeable_actions"] == 3
    assert totals["early_menu_actions"] == 1
    assert totals["early_turn2_actions"] == totals["early_turn3_actions"] == 2
    assert totals["early_turn5_actions"] == 2
    assert totals["early_turn2_observed"] == totals["early_turn3_observed"] == 1
    assert totals["early_turn4_observed"] == 0


@pytest.mark.parametrize("action", [1, 2, 3, NATIVE["A_BUY"], NATIVE["A_SELL"],
    NATIVE["A_PLAY"], NATIVE["A_MOVE"], NATIVE["A_HERO_POWER"],
    NATIVE["A_CHOOSE"], NATIVE["A_CANCEL"]])
def test_all_non_end_action_types_include_menu_clicks(action):
    tracker = shaper()
    assert tracker.step(action, obs(1), obs(1), 0)[0] == -.001
    assert tracker._totals["early_actions"] == tracker._totals["early_chargeable_actions"] == 1


def test_cost_cap_survives_cycle_history_clear_and_cycles_keep_linear_cost():
    tracker = ActionRewardShaper(RewardShapingConfig(**COSTS))
    state = obs(1)
    for index in range(15):
        # Explicit cycle clearing emulates productive/menu-boundary resets; it
        # must not reset the independent early-action ledger or per-game flags.
        tracker._clear_history()
        tracker.step(NATIVE["A_MOVE"], state, state, 0)
    assert tracker._totals["early_action_penalty"] == -.01
    assert tracker._totals["move_penalty"] == pytest.approx(-15 * .0005)
    before = tracker._totals.copy()
    reward, _ = tracker.step(NATIVE["A_MOVE"], state, state, 0)
    assert reward == pytest.approx(-.0105)
    assert tracker._totals["early_action_penalty"] == -.01
    assert tracker._totals["cycle_penalty"] - before["cycle_penalty"] == pytest.approx(-.01)
    assert tracker._totals["early_chargeable_actions"] == 16
    assert tracker._totals["early_turn1_observed"] == 1


def test_terminal_executed_action_costs_but_intercepted_request_does_not():
    tracker = shaper()
    reward, totals = tracker.step(2, obs(1), obs(0), -1, done=True)
    assert reward == pytest.approx(-1.001)
    assert totals["early_actions"] == totals["early_chargeable_actions"] == 1
    assert totals["early_turn1_actions"] == totals["early_turn1_observed"] == 1
    reward, totals = tracker.step(2, obs(1), obs(0), -1, done=True,
                                  terminal_reason="action_limit_forfeit")
    assert reward == -1
    assert totals["early_actions"] == totals["early_chargeable_actions"] == 0
    assert totals["early_action_penalty"] == 0
    assert tracker.step(2, obs(1), obs(1), 0)[0] == -.001  # Fresh episode ledger.


def test_milestones_require_real_same_turn_changes_once_per_episode():
    tracker = shaper()
    empty, full = obs(1), obs(1, board=1)
    # Buying a card into hand is not a board appearance.
    assert tracker.step(NATIVE["A_BUY"], empty, empty, 0)[0] == -.001
    assert tracker.step(NATIVE["A_PLAY"], empty, full, 0)[0] == pytest.approx(.009)
    tracker.step(NATIVE["A_SELL"], full, empty, 0)
    assert tracker.step(NATIVE["A_PLAY"], empty, full, 0)[0] == pytest.approx(-.001)
    one, two = obs(2, tier=1), obs(2, tier=2)
    assert tracker.step(3, one, two, 0)[0] == pytest.approx(.004)
    assert tracker.step(3, one, two, 0)[0] == pytest.approx(-.001)
    totals = tracker.finish_terminal(0, 8)
    assert totals["first_turn_minion_bonus_count"] == totals["second_turn_tier2_bonus_count"] == 1
    assert totals["opening_bonus"] == pytest.approx(.015)
    assert tracker.step(NATIVE["A_HERO_POWER"], empty, full, 0)[0] == pytest.approx(.009)
    tracker.reset()
    assert tracker.step(0, empty, obs(2, board=1), 0)[0] == 0
    assert tracker.step(NATIVE["A_PLAY"], empty, obs(0, board=1), 0, done=True)[1][
        "first_turn_minion_bonus_count"] == 0


@pytest.mark.parametrize("boundary", ["source", "uses", "cancel", "end", "turn", "reset"])
def test_dark_gift_requires_matching_paid_discover_not_arbitrary_choose(boundary):
    tracker = shaper()
    start, pending = obs(3), obs(3, uses=1, pending=2, source=123)
    assert tracker.step(NATIVE["A_DARK_GIFT"], start, pending, 0)[0] == -.0005
    assert tracker._totals["third_turn_dark_gift_bonus"] == 0
    if boundary == "source":
        pending = obs(3, uses=1, pending=2, source=124)
    elif boundary == "uses":
        pending = obs(3, uses=2, pending=2, source=123)
    elif boundary == "cancel":
        tracker.step(NATIVE["A_CANCEL"], pending, start, 0)
    elif boundary == "end":
        tracker.step(0, pending, pending, 0)
    elif boundary == "turn":
        tracker.step(2, obs(4), obs(4), 0)
    else:
        tracker.reset()
    assert tracker.step(NATIVE["A_CHOOSE"], pending, obs(3, uses=1), 0)[0] == pytest.approx(-.0005)
    assert tracker._totals["third_turn_dark_gift_bonus_count"] == 0


def test_dark_gift_transaction_survives_cycle_clear_and_rewards_once():
    tracker = shaper()
    start, pending, picked = obs(3), obs(3, uses=1, pending=2, source=123), obs(3, uses=1)
    tracker.step(NATIVE["A_DARK_GIFT"], start, pending, 0)
    tracker._clear_history()
    assert tracker.step(NATIVE["A_CHOOSE"], pending, picked, 0)[0] == pytest.approx(.0045)
    assert tracker.step(NATIVE["A_CHOOSE"], pending, picked, 0)[0] == pytest.approx(-.0005)
    totals = tracker.finish_terminal(0, 8)
    assert totals["third_turn_dark_gift_bonus_count"] == 1
    assert totals["opening_bonus"] == .005


@pytest.mark.parametrize("costs", [None, [], [0] * 4, [0] * 6, [.001] * 4 + [-1],
    [0] * 4 + [float("nan")], [0] * 4 + [float("inf")]])
def test_invalid_early_costs_rejected_before_sampling(costs):
    with pytest.raises(ValueError, match="early_action_costs"):
        RewardShapingConfig(early_action_costs=costs)


@pytest.mark.parametrize("field", ["early_action_penalty_cap", *OPENING_BONUS_FIELDS])
def test_invalid_early_cap_or_bonus_rejected(field):
    for value in (-.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            RewardShapingConfig(**{field: value})


def test_default_protocol_normalization_is_narrow_and_json_stable():
    old = {"operation_cost": 0, "cycle_cost": .01, "cycle_detection": "all_actions"}
    defaults = {"move_cost": 0, "early_action_costs": (0.,) * 5,
                "early_action_penalty_cap": .01, **dict.fromkeys(OPENING_BONUS_FIELDS, 0)}
    assert reward_protocols_equal(old, {**old, **defaults})
    assert reward_protocols_equal({**old, **defaults}, json.loads(json.dumps({**old, **defaults})))
    assert not reward_protocols_equal(None, {})
    assert not reward_protocols_equal(old, {**old, "cycle_forfeit_repetitions": None})
    assert not reward_protocols_equal(old, {**old, "early_action_costs": [.001] * 5})
    assert not reward_protocols_equal(old, {**old, "first_turn_minion_bonus": .01})
    assert set(old) == {"operation_cost", "cycle_cost", "cycle_detection"}


def native_game(turn=1):
    game = tg.Game(7, json.dumps({"hero_pool": ["TB_BaconShop_HERO_34"], "hero_choices": 1}))
    for _ in range(10000):
        if game.turn >= turn and game.needs_action(0):
            break
        for seat in range(8):
            if game.needs_action(seat):
                game.step(seat, game.bot_action(seat))
    assert game.turn == turn and game.needs_action(0)
    for zone in ("board", "hand", "shop"):
        game.debug(json.dumps({"cmd": "clear_zone", "p": 0, "zone": zone}))
    card = next(c["id"] for c in json.loads(tg.catalog_json()) if c["kind"] == "minion"
                and c["in_pool"] and c["implemented"] and c["tier"] == 1)
    game.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "shop", "card": card}))
    game.debug(json.dumps({"cmd": "set_player", "p": 0, "gold": 10}))
    return game


def native_obs(game):
    features, cards = game.observe(0)
    return {"features": np.frombuffer(features, dtype="<f4").copy(),
            "cards": np.frombuffer(cards, dtype="<i4").copy(),
            "action_mask": np.frombuffer(game.legal_mask(0), dtype=np.uint8).astype(bool)}


def native_step(game, tracker, action):
    before = native_obs(game)
    assert before["action_mask"][action]
    game.step(0, action)
    return tracker.step(action, before, native_obs(game), 0)[0]


def test_native_buy_then_play_and_upgrade_give_completion_bonuses():
    tracker = shaper()
    game = native_game(1)
    assert native_step(game, tracker, NATIVE["A_BUY"]) == -.001
    assert tracker._totals["first_turn_minion_bonus_count"] == 0
    assert native_step(game, tracker, NATIVE["A_PLAY"]) == pytest.approx(.009)
    assert tracker._totals["first_turn_minion_bonus_count"] == 1
    game = native_game(2)
    assert native_step(game, tracker, NATIVE["A_UPGRADE"]) == pytest.approx(.004)
    assert round(float(native_obs(game)["features"][10]) * 6) == 2


class KeepCollecting(BaseCallback):
    def _on_step(self):
        return True


def test_native_dark_gift_completes_across_rollouts_without_adam(monkeypatch):
    torch.set_num_threads(2)
    game = native_game(3)
    env = NativeVectorEnv(2, seed=7, reward_shaping=COSTS)
    model = EntityPPO(EntityPolicy, env, n_steps=1, batch_size=2, n_epochs=1,
        gamma=1, gae_lambda=1, device="cpu", seed=7, policy_kwargs={"model_config": asdict(
            ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})
    calls = []

    def actions(obs, deterministic=False, action_masks=None):
        action = NATIVE["A_DARK_GIFT"] if not calls else NATIVE["A_CHOOSE"]
        calls.append(action)
        selected = torch.full((2,), action, dtype=torch.long)
        assert obs["action_mask"].bool()[torch.arange(2), selected].all()
        return selected, torch.zeros((2, 1)), torch.zeros(2)

    monkeypatch.setattr(model.policy, "forward", actions)
    monkeypatch.setattr(model.policy, "predict_values", lambda obs: torch.zeros((2, 1)))
    before_weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
    try:
        _, callback = model._setup_learn(4, KeepCollecting())
        for lane in range(2):
            state = json.loads(env.batch.save(lane))
            state["game"] = json.loads(game.state_json())
            env.batch.load(lane, json.dumps(state))
        model._last_obs = env._observe()
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 1)
        np.testing.assert_allclose(model.rollout_buffer.rewards, [[-.0005, -.0005]])
        for ledger in env._reward_shapers:
            assert ledger._dark_gift_transaction is not None
            assert ledger._totals["third_turn_dark_gift_bonus_count"] == 0
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 1)
        np.testing.assert_allclose(model.rollout_buffer.rewards, [[.0045, .0045]])
        np.testing.assert_allclose(model.rollout_buffer.returns, [[.0045, .0045]])
        for ledger in env._reward_shapers:
            assert ledger._totals["third_turn_dark_gift_bonus_count"] == 1
            assert ledger._totals["early_turn3_actions"] == 2
            assert ledger._totals["early_action_penalty"] == -.001
        assert model.optimizer_step_count() == 0 and not model.policy.optimizer.state
        assert all(torch.equal(value, before_weights[key])
                   for key, value in model.policy.state_dict().items())
    finally:
        env.close()


def test_native_opening_bonus_and_cost_reach_multi_step_gae_without_adam(monkeypatch):
    game = native_game(1)
    env = NativeVectorEnv(1, seed=7, reward_shaping=COSTS)
    model = EntityPPO(EntityPolicy, env, n_steps=2, batch_size=2, n_epochs=1,
        gamma=1, gae_lambda=1, device="cpu", seed=7, policy_kwargs={"model_config": asdict(
            ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})
    actions = iter((NATIVE["A_BUY"], NATIVE["A_PLAY"]))

    def policy(obs, deterministic=False, action_masks=None):
        action = next(actions)
        assert obs["action_mask"].bool()[0, action]
        return torch.tensor([action]), torch.zeros((1, 1)), torch.zeros(1)

    monkeypatch.setattr(model.policy, "forward", policy)
    monkeypatch.setattr(model.policy, "predict_values", lambda obs: torch.zeros((1, 1)))
    try:
        _, callback = model._setup_learn(2, KeepCollecting())
        state = json.loads(env.batch.save(0))
        state["game"] = json.loads(game.state_json())
        env.batch.load(0, json.dumps(state))
        model._last_obs = env._observe()
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 2)
        np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0], [-.001, .009], atol=1e-8)
        np.testing.assert_allclose(model.rollout_buffer.returns[:, 0], [.008, .009], atol=1e-8)
        np.testing.assert_allclose(model.rollout_buffer.advantages[:, 0], [.008, .009], atol=1e-8)
        assert model.optimizer_step_count() == 0 and not model.policy.optimizer.state
    finally:
        env.close()


def test_unified_actor_dark_gift_and_cost_ledger_survive_collect_boundaries():
    class ChooseActor:
        def __init__(self, actions=None):
            self.actions = iter(actions) if actions else None

        def act(self, observations):
            masks = observations["action_mask"]
            actions = (np.full(len(masks), next(self.actions), dtype=np.int64)
                       if self.actions else masks.argmax(axis=1))
            assert masks[np.arange(len(actions)), actions].all()
            return ActorOutput(actions, np.zeros(len(actions), np.float32),
                               np.zeros(len(actions), np.float32))

    game = native_game(3)
    fixed = FrozenOpponentPool((OpponentGroup("fixed", ChooseActor(), 1, COSTS),))
    sampler = UnifiedActorSampler(1, ChooseActor([NATIVE["A_DARK_GIFT"], NATIVE["A_CHOOSE"]]),
        fixed, frozen_opponents=1, seed=7, threads=1, reward_shaping=COSTS)
    state = json.loads(sampler.batch.save(0))
    state.update(game=json.loads(game.state_json()), seats=[0, 1])
    sampler.batch.load(0, json.dumps(state))
    transitions = []
    sampler.collect(1, transitions.append)
    ledger = sampler.shapers[0]
    assert ledger._totals["early_actions"] == 1 and ledger._dark_gift_transaction is not None
    sampler.collect(1, transitions.append)
    assert sampler.shapers[0] is ledger
    assert [t.reward for t in transitions] == pytest.approx([-.0005, .0045])
    assert ledger._totals["early_action_penalty"] == -.001
    assert ledger._totals["early_turn3_actions"] == 2
    assert ledger._totals["third_turn_dark_gift_bonus_count"] == 1


def test_zero_opening_defaults_preserve_native_state_masks_rewards_and_placements():
    legacy = {"operation_cost": 0, "cycle_cost": .01, "cycle_detection": "all_actions"}
    explicit = {**legacy, "early_action_costs": [0] * 5,
                **dict.fromkeys(OPENING_BONUS_FIELDS, 0), "early_action_penalty_cap": .01}
    left, right = [NativeVectorEnv(1, seed=71, threads=1, config={"turn_action_caps": [3]},
                  reward_shaping=costs) for costs in (legacy, explicit)]
    a, b = left.reset(), right.reset()
    ended = 0
    try:
        for _ in range(100):
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
            actions = a["action_mask"].argmax(axis=1)
            a, ar, ad, ai = left.step(actions)
            b, br, bd, bi = right.step(actions)
            np.testing.assert_array_equal(ar, br)
            np.testing.assert_array_equal(ad, bd)
            assert left.batch.save(0) == right.batch.save(0)
            if ad[0]:
                ended += 1
                assert ai[0]["placement"] == bi[0]["placement"]
                assert ai[0]["reward_components"] == bi[0]["reward_components"]
        assert ended > 0
    finally:
        left.close()
        right.close()


def test_native_rejected_action_never_enters_opening_ledger():
    env = NativeVectorEnv(1, seed=7, reward_shaping=COSTS)
    state = env.reset()
    invalid = int(np.flatnonzero(~state["action_mask"][0])[0])
    try:
        with pytest.raises(RuntimeError, match="rejected"):
            env.step([invalid])
        ledger = env._reward_shapers[0]
        assert ledger._totals["early_actions"] == ledger._totals["early_action_penalty"] == 0
    finally:
        env.close()


def test_native_forced_end_counts_without_opening_cost_and_continues_game():
    env = NativeVectorEnv(1, seed=7, config={"turn_action_caps": [3]}, reward_shaping=COSTS)
    state = env.reset()
    try:
        for _ in range(10):
            if round(float(state["features"][0, 0]) * 20) == 1:
                break
            state, _, _, _ = env.step([int(state["action_mask"][0].argmax())])
        for _ in range(3):
            assert state["action_mask"][0, 2]
            state, _, _, _ = env.step([2])
        assert state["action_mask"][0].sum() == 1 and state["action_mask"][0, 0]
        ledger = env._reward_shapers[0]
        penalty = ledger._totals["early_action_penalty"]
        state, reward, done, _ = env.step([0])
        assert not done[0] and reward[0] == 0
        assert round(float(state["features"][0, 0]) * 20) == 2
        assert ledger._totals["early_action_penalty"] == penalty == -.003
        assert ledger._totals["early_turn1_actions"] == 4
        assert ledger._totals["early_chargeable_actions"] == 3
    finally:
        env.close()


def test_opening_metrics_map_to_train_val_greedy_and_mock_journal(tmp_path, monkeypatch):
    recorded = []
    run = SimpleNamespace(entity="eric_eric_", project="AutoTavern", id="mock", url="mock",
        define_metric=lambda *a, **k: None, log=recorded.append, finish=lambda **k: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **k: run,
                                                             Settings=lambda **k: k))
    tracker = WandbTracker(tmp_path, {"tracking": {"mode": "offline"}})
    metrics = {key: float(index) for index, key in enumerate(REWARD_METRICS)}
    try:
        tracker.log({f"rollout/{key}": value for key, value in metrics.items()}, 0)
        values = {}
        logger = SimpleNamespace(record=lambda key, value: values.update({key: value}))
        result = {"conditions": {"mixed_sampled": metrics, "mixed_greedy": metrics}}
        result["conditions"] = {
            name: {**row, "mean_placement": 3., "first_rate": .2, "top4_rate": .7}
            for name, row in result["conditions"].items()}
        log_evaluation(logger, "val", result)
        tracker.log(values, 0)
    finally:
        tracker.finish()
    rows = [json.loads(line) for line in
            (tmp_path / "wandb-metrics.jsonl").read_text().splitlines()]
    assert rows == recorded
    for key in ("early_actions", "early_chargeable_actions", "early_menu_actions",
                "early_action_penalty", "early_turn1_actions", "early_turn5_observed",
                "opening_bonus", *OPENING_BONUS_FIELDS):
        assert f"train/{key}" in rows[0]
        assert f"val/{key}" in rows[1] and f"val/greedy_{key}" in rows[1]
