"""League timing, native equivalence, exact recovery and opponent visibility."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import taverngym as tg
from training.league import LeagueEnv


def league(n=4, *, threads=1, ids=None, opponents=None, guard=True):
    return LeagueEnv(n, seed=17, opponents=opponents or [None], opponent_ids=ids or ["bot"],
                     device="cpu", threads=threads, config={"turn_action_caps": [32]},
                     cancel_guard=guard)


def bot_actions(env):
    obs = env.observe()
    actions = []
    for lane, seat in enumerate(env.learners):
        action = env.game(lane).bot_action(seat)
        actions.append(action if obs["action_mask"][lane, action]
                       else np.flatnonzero(obs["action_mask"][lane])[0])
    return actions


def test_single_lane_matches_native_multi_seat_driver():
    env = league(1, guard=False)
    native = tg.BatchEnv(1, seed=17, controlled=tuple(range(8)),
                         config={"turn_action_caps": [32]})
    try:
        assert env.game(0).state_json() == native.game(0).state_json()
        for _ in range(1000):
            action = bot_actions(env)
            lr, ld, li = env.step(action)
            for tick in range(1000):
                obs = native.observe()
                game = native.game(0)
                actions = [game.bot_action(s) if obs["ready"][s] else 0 for s in range(8)]
                if tick == 0:
                    actions[0] = action[0]
                else:
                    assert not obs["ready"][0], "a ready learner must be held unchanged"
                nr, nd, ni = native.step(actions)
                if nd[0] or native.observe()["ready"][0]:
                    break
            assert ld[0] == nd[0]
            np.testing.assert_array_equal(lr, nr[:1])
            if ld[0]:
                assert li["placement"][0] == ni["placement"][0]
                assert env.episode_index(0) == 1
                break
            assert env.game(0).state_json() == native.game(0).state_json()
        else:
            pytest.fail("No learner episode completed")
    finally:
        env.close()


def test_restore_and_thread_count_preserve_games_masks_and_rewards():
    a, b = league(threads=1), league(threads=2)
    try:
        for _ in range(75):
            a.step(bot_actions(a))
        for lane in range(a.n_games):
            b.load(lane, a.save(lane))
        for _ in range(180):
            oa, ob = a.observe(), b.observe()
            for key in oa:
                np.testing.assert_array_equal(oa[key], ob[key])
            actions = bot_actions(a)
            ar, ad, ai = a.step(actions)
            br, bd, bi = b.step(actions)
            for left, right in [(ar, br), (ad, bd), (ai["placement"], bi["placement"])]:
                np.testing.assert_array_equal(left, right)
        assert sum(a.indices) > 0
        assert [a.save(i) for i in range(4)] == [b.save(i) for i in range(4)]
    finally:
        a.close(); b.close()


class VisibleEndTurn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features, cards, masks):
        self.calls += 1
        assert features.shape[1] == tg.OBS_FLOATS and cards.shape[1] == tg.OBS_CARDS
        assert masks.any(axis=1).all()
        return torch.zeros_like(masks, dtype=torch.float32).masked_fill(~masks, -1e9), torch.zeros(len(masks))


def test_frozen_opponents_receive_only_ready_player_observations():
    policy = VisibleEndTurn()
    env = league(opponents=[None, policy], ids=["bot", "frozen"])
    completed = 0
    try:
        for _ in range(180):
            rewards, done, info = env.step(bot_actions(env))
            assert env.observe()["ready"].all()
            assert np.all((info["placement"][done] >= 1) & (info["placement"][done] <= 8))
            assert np.all(np.abs(rewards) <= 1)
            completed += done.sum()
        assert policy.calls > 0 and completed > 0
    finally:
        env.close()


def test_changed_opponent_pool_and_invalid_actions_are_rejected():
    a, b = league(), league(ids=["different-checkpoint-hash"])
    try:
        with pytest.raises(ValueError, match="opponent_ids"):
            b.load(0, a.save(0))
        before = [a.save(i) for i in range(4)]
        with pytest.raises(ValueError, match="out of range"):
            a.step([999] * 4)
        # Observation preparation may initialize decoder state; game state stays exact.
        import json
        assert [json.loads(a.save(i))["game"] for i in range(4)] == [json.loads(s)["game"] for s in before]
    finally:
        a.close(); b.close()
