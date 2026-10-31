"""The batched environment (`taverngym.BatchEnv`)."""

import json

import numpy as np
import pytest

import taverngym
from taverngym import N_ACTIONS, OBS_CARDS, OBS_FLOATS, BatchEnv


def random_actions(obs, rng):
    """A random legal action per ready slot, ending the turn now and then."""
    out = np.zeros(len(obs["ready"]), dtype=np.int64)
    for s in np.flatnonzero(obs["ready"]):
        legal = np.flatnonzero(obs["action_mask"][s])
        out[s] = 0 if legal[0] == 0 and rng.random() < 0.3 else rng.choice(legal)
    return out


def same(h1, h2):
    return len(h1) == len(h2) and all(
        np.array_equal(u, v) for x, y in zip(h1, h2) for u, v in zip(x, y))


def run(env, steps, seed=0):
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(steps):
        obs = env.observe()
        rewards, dones, info = env.step(random_actions(obs, rng))
        history.append((rewards.copy(), dones.copy(), info["placement"].copy()))
    return history


def test_shapes_and_slots():
    env = BatchEnv(3, seed=1, controlled=(0, 5), threads=2)
    assert (env.n_games, env.n_slots, env.controlled) == (3, 6, (0, 5))
    assert [env.slot(i) for i in range(6)] == [(0, 0), (0, 5), (1, 0), (1, 5), (2, 0), (2, 5)]
    obs = env.observe()
    assert obs["features"].shape == (6, OBS_FLOATS) and obs["features"].dtype == np.float32
    assert obs["cards"].shape == (6, OBS_CARDS) and obs["cards"].dtype == np.int32
    assert obs["action_mask"].shape == (6, N_ACTIONS) and obs["ready"].shape == (6,)
    # A ready slot has at least one legal action; a waiting slot none.
    for s in range(6):
        assert obs["action_mask"][s].any() == obs["ready"][s]


def test_episodes_end_and_games_restart():
    env = BatchEnv(2, seed=3, controlled=(0, 1, 2), threads=2)
    history = run(env, 1500)
    places = [p for _, dones, placement in history for p, d in zip(placement, dones) if d]
    assert places and all(1 <= p <= 8 for p in places)
    rewards = np.concatenate([r for r, _, _ in history])
    assert rewards.min() >= -1.0 and rewards.max() <= 1.0
    assert max(env.episode_index(i) for i in range(2)) >= 1, "a finished game restarts"


def test_threads_do_not_change_results():
    a = run(BatchEnv(4, seed=7, controlled=(0,), threads=1), 300)
    b = run(BatchEnv(4, seed=7, controlled=(0,), threads=4), 300)
    assert same(a, b)


def test_rejected_actions_are_ignored():
    env = BatchEnv(1, seed=2)
    before = env.save(0)
    rewards, dones, info = env.step([N_ACTIONS + 5])
    assert info["rejected"][0] and rewards[0] < 0 and not dones[0]
    assert not info["deferred"].any()
    assert env.save(0) == before
    with pytest.raises(ValueError):
        env.step([0, 0])


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("existing_discover", [False, True])
def test_recruit_death_defers_bigglesworth_stale_move_or_discover(threads, existing_discover):
    """An earlier seat dies after every actor has selected an initially legal action."""
    cfg = {"hero_pool": ["TB_BaconShop_HERO_34"], "hero_choices": 1}
    game = taverngym.Game(11, json.dumps(cfg))
    for _ in range(100):
        if game.turn >= 1:
            break
        for seat in range(8):
            if game.needs_action(seat):
                game.step(seat, game.bot_action(seat))
    for seat, hero in ((0, "TB_BaconShop_HERO_25"), (1, "TB_BaconShop_HERO_70")):
        game.debug(json.dumps({"cmd": "set_hero", "p": seat, "hero": hero}))
        for zone in ("board", "hand", "shop"):
            game.debug(json.dumps({"cmd": "clear_zone", "p": seat, "zone": zone}))
    card = next(c["id"] for c in json.loads(taverngym.catalog_json())
                if c["kind"] == "minion" and c["in_pool"] and c["implemented"])
    for seat in (0, 1, 1):
        game.debug(json.dumps({"cmd": "add_minion", "p": seat, "zone": "board", "card": card}))
    game.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "shop", "card": card}))
    game.debug(json.dumps({"cmd": "set_player", "p": 0, "health": 1, "armor": 0, "gold": 10}))
    game.step(0, 4)  # Baz'hial's target selection; resolving it will kill her.
    state = json.loads(game.state_json())
    if existing_discover:
        state["players"][1]["pending"] = [{"Discover": {
            "source": state["players"][1]["powers"][0]["card"],
            "options": [{"Minion": state["players"][1]["board"][0]}],
        }}]
    env = BatchEnv(1, controlled=(0, 1), threads=threads, config=cfg)
    episode = json.loads(env.save(0))
    episode["game"] = state
    env.load(0, json.dumps(episode))
    action = 30 if existing_discover else 203
    before = env.observe()
    assert before["action_mask"][[0, 1], [30, action]].all()
    old_board = state["players"][1]["board"]
    rewards, dones, info = env.step([30, action])
    assert not info["rejected"].any()
    np.testing.assert_array_equal(info["deferred"], [False, True])
    np.testing.assert_array_equal(dones, [True, False])
    np.testing.assert_array_equal(rewards, [-1, 0])
    state = json.loads(env.save(0))["game"]
    assert state["players"][1]["board"] == old_board
    assert len(state["players"][1]["pending"]) == 1 + int(existing_discover)
    # The fresh Discover is answered using its new observation, without any
    # illegal-action penalty or unrecorded stand-in operation.
    after = env.observe()
    assert after["action_mask"][1, 30]
    _, _, info = env.step([0, 30])
    assert not info["rejected"].any() and not info["deferred"].any()


def test_save_and_load_continue_exactly():
    a = BatchEnv(1, seed=5)
    run(a, 40, seed=1)
    b = BatchEnv(1, seed=5)  # fresh; the same seed for the games after this one
    b.load(0, a.save(0))
    assert same(run(a, 60, seed=2), run(b, 60, seed=2))
    assert a.save(0) == b.save(0)
    bad = json.loads(a.save(0))
    bad["game"]["card_table"] ^= 1
    with pytest.raises(ValueError, match="card table"):
        b.load(0, json.dumps(bad))


def test_a_game_copy_can_be_inspected():
    env = BatchEnv(2, seed=4)
    g = env.game(1)
    assert "me" in json.loads(g.view(0))
    assert isinstance(g, taverngym.Game)
    assert json.loads(g.state_json())["turn"] == json.loads(env.save(1))["game"]["turn"]


def test_bad_arguments():
    for kwargs in (dict(n_games=0), dict(n_games=1, controlled=()),
                   dict(n_games=1, controlled=(8,)), dict(n_games=1, controlled=(1, 1))):
        with pytest.raises(ValueError):
            BatchEnv(**kwargs)
    with pytest.raises(ValueError, match="hero_choice"):
        BatchEnv(1, config={"hero_choice": 4})


def test_active_games_hold_every_other_state_exactly():
    env = BatchEnv(3, seed=9, controlled=(0, 1, 2, 3, 4), rotate_seats=True, threads=3)
    before = [env.save(i) for i in range(3)]
    obs = env.observe()
    actions = random_actions(obs, np.random.default_rng(5))
    actions[5:] = N_ACTIONS + 5  # Even illegal actions in held games must be ignored.
    rewards, dones, info = env.step(actions, active=[True, False, False])
    assert env.save(0) != before[0]
    assert [env.save(i) for i in (1, 2)] == before[1:]
    assert not info["rejected"].any()
    assert not rewards[5:].any() and not dones[5:].any()
    with pytest.raises(ValueError, match="one flag"):
        env.step(actions, active=[True])


def test_rotated_roles_cover_seats_and_replay_across_threads():
    a = BatchEnv(16, seed=9, controlled=range(5), rotate_seats=True, threads=1)
    b = BatchEnv(16, seed=9, controlled=range(5), rotate_seats=True, threads=3)
    seats = a.slot_seats.reshape(16, 5)
    assert set(seats[:, 0]) == set(range(8))
    assert all(len(set(row)) == 5 for row in seats)
    np.testing.assert_array_equal(a.slot_seats, b.slot_seats)
    for i in range(16):
        b.load(i, a.save(i))
    assert same(run(a, 120, seed=5), run(b, 120, seed=5))
    assert any(a.episode_index(i) > 0 for i in range(16))
    np.testing.assert_array_equal(a.slot_seats, b.slot_seats)
    for i in range(16):
        assert a.save(i) == b.save(i)
