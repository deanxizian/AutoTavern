"""Independent scheduling must preserve each learner's complete trajectory."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from taverngym.learning.actor_sampling import ActorOutput, UnifiedActorSampler
from taverngym.learning.vector_env import MixedOpponentVectorEnv
from taverngym.learning.opponents import FrozenOpponentPool, OpponentGroup


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
def test_independent_games_match_synchronous_trajectories_across_two_rollouts(threads, frozen_opponents):
    options = dict(seed=12, threads=threads, config={"turn_action_caps": [6]}, reward_shaping={})
    learner, fixed = RuleActor(), RuleActor(slow=True)
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
    sampler = UnifiedActorSampler(3, RuleActor(), actor_fixed,
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
        assert env.waiting_rounds > 0
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
