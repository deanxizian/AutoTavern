"""The optional filter must stop real native cycles without changing game legality."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import taverngym as tg
from taverngym.learning.action_filter import ReversibleActionFilter


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
    for zone in ("board", "hand", "shop"):
        g.debug(json.dumps({"cmd": "clear_zone", "p": 0, "zone": zone}))
    cards = [c["id"] for c in json.loads(tg.catalog_json()) if c["kind"] == "minion" and c["in_pool"] and c["implemented"]][:3]
    for card in cards:
        g.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "board", "card": card}))
    g.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "shop", "card": cards[0]}))
    g.debug(json.dumps({"cmd": "set_player", "p": 0, "gold": 10}))
    return g


def execute(g, guard, action):
    before = observation(g)
    assert guard.mask(before)[action]
    g.step(0, action)
    after = observation(g)
    guard.commit(action, before, after)
    return after


def test_inverse_move_and_longer_cycle_are_blocked(game):
    guard = ReversibleActionFilter()
    after = execute(game, guard, 196 + 0 * 7 + 1)
    reverse = 196 + 1 * 7
    assert after["action_mask"][reverse]  # still a legal game action
    assert not guard.mask(after)[reverse]
    after = execute(game, guard, 196 + 1 * 7 + 2)
    assert not guard.mask(after)[196 + 2 * 7]  # ABC -> BAC -> BCA -> ABC
    assert guard.mask(after)[0]  # EndTurn is retained


def test_freeze_roundtrip_blocked_and_refresh_resets_it(game):
    guard = ReversibleActionFilter()
    after = execute(game, guard, 2)
    assert after["action_mask"][2] and not guard.mask(after)[2]
    after = execute(game, guard, 1)
    assert guard.mask(after)[2]


def test_freeze_does_not_erase_move_history(game):
    guard = ReversibleActionFilter()
    execute(game, guard, 197)
    after = execute(game, guard, 2)
    assert not guard.mask(after)[203]


def test_viewing_recommendations_is_idempotent_and_inputs_unchanged(game):
    guard = ReversibleActionFilter()
    obs = observation(game)
    copy = {key: value.copy() for key, value in obs.items()}
    first = guard.mask(obs)
    for _ in range(5):
        np.testing.assert_array_equal(guard.mask(obs), first)
    assert guard.reversible_actions == 0
    for key in obs:
        np.testing.assert_array_equal(obs[key], copy[key])
    assert not (first & ~obs["action_mask"]).any()
    native = obs["action_mask"]
    # Only Move and Freeze may be removed; spending actions and EndTurn remain.
    np.testing.assert_array_equal(first[:2], native[:2])
    np.testing.assert_array_equal(first[3:196], native[3:196])


def test_budget_and_episode_reset_keep_productive_actions_available(game):
    guard = ReversibleActionFilter(max_reversible_actions=1)
    after = execute(game, guard, 197)
    allowed = guard.mask(after)
    assert not allowed[196:].any() and not allowed[2]
    assert allowed[0] and allowed[1] and allowed[8]
    guard.commit(0, after, after, done=True)
    assert guard.mask(after)[203]
