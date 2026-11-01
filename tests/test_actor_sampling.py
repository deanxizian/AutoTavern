"""Independent scheduling must preserve each learner's complete trajectory."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from taverngym.learning.actor_sampling import ActorOutput, UnifiedActorSampler
from taverngym.learning.vector_env import MixedOpponentVectorEnv
from taverngym.learning.opponents import FrozenOpponentPool, OpponentGroup
import taverngym as tg


class RuleActor:
    def __init__(self, slow=False):
        self.slow = slow
        self.batches = []

    def act(self, obs):
        mask = obs["action_mask"]
        actions = mask.argmax(axis=1)
        # Distinct policies and different turn lengths exercise waiting lanes.
        freeze = mask[:, 2] & (self.slow | (obs["features"][:, 15] == 0))
        actions[freeze] = 2
        self.batches.append(len(actions))
        return ActorOutput(actions, np.zeros(len(actions), np.float32), np.zeros(len(actions), np.float32))

    def predict(self, obs, deterministic=False):
        return self.act(obs).actions, None


def copy_obs(obs, row=None):
    return {key: (value if row is None else value[row]).copy() for key, value in obs.items()}


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("frozen_opponents", [1, 2, 3, 5])
@pytest.mark.parametrize("forfeit", [False, "cycle", "action_limit"])
def test_independent_games_match_synchronous_trajectories_across_two_rollouts(
        threads, frozen_opponents, forfeit):
    costs = ({"operation_cost": 0, "cycle_cost": 0.0002, "cycle_detection": "all_actions",
              "cycle_forfeit_repetitions": 3 if forfeit == "cycle" else None,
              "turn_action_forfeit_limit": 4 if forfeit == "action_limit" else None}
             if forfeit else {})
    options = dict(seed=12, threads=threads, config={"turn_action_caps": [60 if forfeit else 6]},
                   reward_shaping=costs)
    learner, fixed = RuleActor(slow=bool(forfeit)), RuleActor(slow=True)
    actor_fixed = RuleActor(slow=True)
    if frozen_opponents == 5:
        fixed = FrozenOpponentPool((OpponentGroup("bot_v1", RuleActor(True), 2),
                                    OpponentGroup("bot_v2", RuleActor(), 3)))
        actor_fixed = FrozenOpponentPool((OpponentGroup("bot_v1", RuleActor(True), 2),
                                          OpponentGroup("bot_v2", RuleActor(), 3)))
    elif frozen_opponents == 3:
        fixed = FrozenOpponentPool(tuple(OpponentGroup(f"snapshot_{i}", RuleActor(i % 2 == 0), 1)
                                         for i in range(3)))
        actor_fixed = FrozenOpponentPool(tuple(OpponentGroup(f"snapshot_{i}", RuleActor(i % 2 == 0), 1)
                                               for i in range(3)))
    env = MixedOpponentVectorEnv(3, opponent_policy=fixed, frozen_opponents=frozen_opponents, **options)
    sampler = UnifiedActorSampler(3, RuleActor(slow=bool(forfeit)), actor_fixed,
                                  frozen_opponents=frozen_opponents, **options)
    expected = [[] for _ in range(3)]
    actual = [[] for _ in range(3)]
    episodes, episode_steps = [0] * 3, [0] * 3
    obs = env.reset()
    for key, value in sampler.learner_observation().items():
        np.testing.assert_array_equal(obs[key], value)
    try:
        for _ in range(180):
            before = obs
            actions = learner.act(before).actions
            obs, rewards, dones, infos = env.step(actions)
            for game in range(3):
                expected[game].append((copy_obs(before, game), int(actions[game]), float(rewards[game]),
                    bool(dones[game]), int(infos[game].get("placement", 0)), copy_obs(obs, game),
                    episodes[game], episode_steps[game]))
                episode_steps[game] += 1
                if dones[game]:
                    episodes[game] += 1
                    episode_steps[game] = 0

        def capture(transition):
            assert isinstance(transition.log_prob, np.float32)
            assert isinstance(transition.value, np.float32)
            assert transition.observation["action_mask"][transition.action]
            assert transition.next_observation["action_mask"].any()
            actual[transition.game].append((transition.observation, transition.action, float(transition.reward),
                transition.done, transition.placement, copy_obs(transition.next_observation),
                transition.episode, transition.episode_step))

        blocks = [sampler.collect(70, capture), sampler.collect(110, capture)]
        assert sum(block["learner_samples"] for block in blocks) == 540
        assert sum(block["terminal_transitions"] for block in blocks) >= 6
        if not forfeit:
            assert env.waiting_rounds > 0
        else:
            assert all(shaper._totals["operation_penalty"] == 0 for shaper in sampler.shapers)
        for game in range(3):
            assert len(actual[game]) == len(expected[game]) == 180
            for want, got in zip(expected[game], actual[game]):
                assert want[1:5] == got[1:5]
                assert want[6:] == got[6:]
                for index in (0, 5):
                    for key, value in want[index].items():
                        np.testing.assert_array_equal(value, got[index][key])
            assert sampler.batch.save(game) == env.batch.save(game)
    finally:
        env.close()


def test_role_aware_moves_charge_once_per_executed_seat_across_rollouts():
    costs = {"operation_cost": 0, "cycle_cost": .01, "cycle_detection": "all_actions",
             "move_cost": .0005}

    class MoveActor(RuleActor):
        def act(self, obs):
            actions = np.full(len(obs["features"]), 197, np.int64)
            assert obs["action_mask"][np.arange(len(actions)), actions].all()
            return ActorOutput(actions, np.zeros(len(actions), np.float32),
                               np.zeros(len(actions), np.float32))

    game = tg.Game(11, json.dumps({"hero_pool": ["TB_BaconShop_HERO_34"], "hero_choices": 1}))
    while game.turn < 1:
        for seat in range(8):
            if game.needs_action(seat):
                game.step(seat, game.bot_action(seat))
    cards = [c["id"] for c in json.loads(tg.catalog_json())
             if c["kind"] == "minion" and c["in_pool"] and c["implemented"]][:2]
    for seat in (0, 1):
        game.debug(json.dumps({"cmd": "clear_zone", "p": seat, "zone": "board"}))
        for card in cards:
            game.debug(json.dumps({"cmd": "add_minion", "p": seat, "zone": "board", "card": card}))
    fixed = FrozenOpponentPool((OpponentGroup("frozen", MoveActor(), 1, costs),))
    sampler = UnifiedActorSampler(1, MoveActor(), fixed, seed=11, threads=1,
                                  frozen_opponents=1, reward_shaping=costs)
    state = json.loads(sampler.batch.save(0))
    state.update(game=json.loads(game.state_json()), seats=[0, 1])
    sampler.batch.load(0, json.dumps(state))
    transitions = []
    sampler.collect(1, transitions.append)
    sampler.collect(1, transitions.append)
    assert [t.reward for t in transitions] == pytest.approx([-.0005, -.0105])
    assert not any(t.done for t in transitions)
    for shaper in (sampler.shapers[0], sampler.opponent_shapers[1]):
        assert shaper._totals["move_actions"] == 2
        assert shaper._totals["move_penalty"] == pytest.approx(-.001)
        assert shaper._totals["cycle_returns"] == 1
