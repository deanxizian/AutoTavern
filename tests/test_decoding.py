"""A cancellation remains possible; repeating its restored opener does not."""

import json

import numpy as np
import pytest

from training.decoding import CancelRetryGuard


def observation(*, turn=1, target=False, gold=3):
    mask = np.zeros((1, 245), bool)
    mask[0, [7, 30] if target else [0, 1, 4]] = True
    return {"features": np.array([[turn / 20, gold, float(target)]], np.float32),
            "cards": np.array([[123]], np.int32), "action_mask": mask,
            "ready": np.array([True])}


def cancelled(guard):
    guard.apply(observation())
    guard.record([4])
    target = guard.apply(observation(target=True))
    assert target["action_mask"][0, 7]
    guard.record([7])


def test_cancel_restores_state_but_cannot_reopen_same_targeting():
    guard = CancelRetryGuard()
    cancelled(guard)
    original = observation()
    prepared = guard.apply(original)
    assert original["action_mask"][0, 4], "native legality must not be mutated"
    assert not prepared["action_mask"][0, 4]
    assert prepared["action_mask"][0, 0]
    assert np.all(prepared["action_mask"] <= original["action_mask"])
    with pytest.raises(ValueError, match="not allowed"):
        guard.record([4])
    guard.record([0])


def test_changed_state_or_new_turn_is_not_the_cancelled_state():
    for changed in [observation(gold=4), observation(turn=2)]:
        guard = CancelRetryGuard()
        cancelled(guard)
        assert guard.apply(changed)["action_mask"][0, 4]


def test_save_load_preserves_pending_cancel_and_earlier_bans():
    guard = CancelRetryGuard()
    cancelled(guard)
    restored = CancelRetryGuard()
    restored.load(0, json.loads(json.dumps(guard.save(0))))
    np.testing.assert_array_equal(restored.apply(observation())["action_mask"],
                                  guard.apply(observation())["action_mask"])
    restored.load(0, json.loads(json.dumps(guard.save(0))))
    assert not restored.apply(observation())["action_mask"][0, 4]
    restored.reset(0)
    assert restored.apply(observation())["action_mask"][0, 4]


def test_matching_state_in_another_lane_is_independent():
    guard = CancelRetryGuard()
    cancelled(guard)
    assert guard.apply(observation(), identities=[1])["action_mask"][0, 4]
    assert not guard.apply(observation(), identities=[0])["action_mask"][0, 4]


def test_tavish_target_cancel_cycle_on_the_real_engine():
    tg = pytest.importorskip("taverngym")
    game = tg.Game(7)
    for seat in range(8):
        game.step(seat, 30)
    for seat in range(8):
        for _ in range(16):
            legal = np.frombuffer(game.legal_mask(seat), np.uint8).astype(bool)
            if legal[0] or not legal.any():
                break
            game.step(seat, int(np.flatnonzero(legal)[0]))
    game.debug(json.dumps({"cmd": "set_hero", "p": 0, "hero": "BG22_HERO_000"}))

    def obs():
        f, c = game.observe(0)
        return {"features": np.frombuffer(f, "<f4")[None, :],
                "cards": np.frombuffer(c, "<i4")[None, :],
                "action_mask": np.frombuffer(game.legal_mask(0), np.uint8).astype(bool)[None, :],
                "ready": np.array([True])}

    guard = CancelRetryGuard()
    before = obs()
    assert guard.apply(before)["action_mask"][0, 4]
    guard.record([4]); game.step(0, 4)
    assert guard.apply(obs())["action_mask"][0, 7]
    guard.record([7]); game.step(0, 7)
    after = obs()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert not guard.apply(after)["action_mask"][0, 4]
