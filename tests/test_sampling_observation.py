"""Private sampling views preserve public bytes and never reuse exposed owners."""
import numpy as np

import taverngym as tg


def test_sampling_transport_matches_public_observations_without_mutating_worlds():
    env = tg.BatchEnv(2, seed=92300001, controlled=tuple(range(7)), threads=2,
                      config={"hero_choices": 2, "turn_action_caps": [60]})
    for _ in range(12):
        before = [env.save(game) for game in range(env.n_games)]
        public = env.observe()
        sampled = env._observe_sampling()
        for key in public:
            np.testing.assert_array_equal(sampled[key], public[key])
            assert sampled[key].flags.writeable
            assert sampled[key].flags.c_contiguous
        assert before == [env.save(game) for game in range(env.n_games)]
        actions = np.array([np.flatnonzero(mask)[0] if mask.any() else 0
                            for mask in sampled["action_mask"]], dtype=np.int64)
        env.step(actions)


def test_sampling_owners_remain_independent_across_calls_and_python_mutation():
    env = tg.BatchEnv(1, seed=92300002, controlled=(0,), threads=1,
                      config={"hero_choices": 2, "turn_action_caps": [60]})
    first = env._observe_sampling()
    snapshots = {key: value.copy() for key, value in first.items()}
    second = env._observe_sampling()
    for key in first:
        assert not np.shares_memory(first[key], second[key])
        second[key].fill(0)
        np.testing.assert_array_equal(first[key], snapshots[key])
    public = env.observe()
    for key in first:
        np.testing.assert_array_equal(public[key], snapshots[key])
