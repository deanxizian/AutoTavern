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


def test_single_lane_matches_seat_ordered_game_reference():
    env = league(1, guard=False)
    reference = tg.BatchEnv(1, seed=17, controlled=tuple(range(8)),
                            config={"turn_action_caps": [32]}).game(0)
    try:
        assert env.game(0).state_json() == reference.state_json()
        for _ in range(1000):
            action = bot_actions(env)
            lr, ld, li = env.step(action)
            reference.step(0, action[0])
            for offset in range(1, 8000):
                seat = offset % 8
                if reference.placements()[0] or (seat == 0 and reference.needs_action(0)):
                    break
                if reference.needs_action(seat):
                    reference.step(seat, reference.bot_action(seat))
            place = reference.placements()[0]
            assert ld[0] == bool(place)
            if ld[0]:
                assert li["placement"][0] == place
                assert lr[0] == np.float32(1) - np.float32(2) * np.float32(place - 1) / np.float32(7)
                assert env.episode_index(0) == 1
                break
            assert lr[0] == 0
            assert env.game(0).state_json() == reference.state_json()
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


def death_discover_game():
    """Pay Health for a shop spell: another seat's Bigglesworth gets a Discover."""
    import json
    game = tg.Game(1)
    for seat in range(8):
        game.step(seat, 30)
    for seat in range(8):
        while json.loads(game.view(seat))["me"]["pending"] is not None:
            game.step(seat, game.bot_action(seat))
        game.debug(json.dumps({"cmd": "set_hero", "p": seat, "hero": "TB_BaconShop_HERO_35"}))
        game.debug(json.dumps({"cmd": "clear_zone", "p": seat, "zone": "hand"}))
    game.debug(json.dumps({"cmd": "set_hero", "p": 6, "hero": "TB_BaconShop_HERO_70"}))
    game.debug(json.dumps({"cmd": "set_player", "p": 2, "health": 1, "armor": 0, "gold": 0}))
    game.debug(json.dumps({"cmd": "add_minion", "p": 2, "zone": "board", "card": "BG28_300"}))
    game.debug(json.dumps({"cmd": "clear_zone", "p": 2, "zone": "shop"}))
    game.debug(json.dumps({"cmd": "add_spell", "p": 2, "zone": "shop", "card": "BG28_571"}))
    assert game.legal_mask(6)[0] and game.bot_action(2) == 8
    return game


def test_cross_player_discover_is_observed_before_learning_action():
    env = league(1)
    try:
        env.games[0] = death_discover_game()
        env.learners[0], env.cursors[0] = 6, 2
        env.assignments[0] = [0] * 8
        env.assignments[0][6] = -1
        env._drive([0])
        assert env.games[0].placements()[2] == 8
        obs = env.observe()
        assert not obs["action_mask"][0, 0], "EndTurn became stale when Bigglesworth received a Discover"
        assert obs["action_mask"][0, 30:46].any()
        env.step([int(np.flatnonzero(obs["action_mask"][0])[0])])
    finally:
        env.close()


def test_arena_reads_cross_player_discover_at_execution_time():
    from training.turns import RoundRobinArena
    env = RoundRobinArena(1, seed=1, controlled=range(8), threads=1, config={})
    env.games[0], env.cursors[0] = death_discover_game(), 2
    try:
        for expected_seat in (2, 3, 4, 5, 6):
            obs = env.observe()
            np.testing.assert_array_equal(np.flatnonzero(obs["ready"]), [expected_seat])
            action = env.game(0).bot_action(expected_seat)
            if expected_seat == 6:
                assert not obs["action_mask"][6, 0]
                assert 30 <= action < 46
            actions = np.zeros(8, np.int64)
            actions[expected_seat] = action
            env.step(actions)
    finally:
        env.close()


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
