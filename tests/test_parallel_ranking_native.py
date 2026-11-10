"""Native terminal/reset contract used by the development ranking scheduler."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import taverngym as tg
from taverngym.learning import parallel_ranking as ranking


def test_native_lobby_reports_all_seats_once_before_auto_reset():
    env = tg.BatchEnv(2, seed=60000007, controlled=range(7), threads=1,
                      config={"turn_action_caps": [60]}, rotate_seats=True)
    initial_seats = env.slot_seats.copy()
    active = np.ones(2, dtype=bool)
    placements = np.zeros((2, 7), dtype=int)
    for _ in range(1000):
        obs = env.observe()
        ready = obs["ready"] & np.repeat(active, 7)
        actions = np.zeros(env.n_slots, dtype=np.int64)
        for slot in np.flatnonzero(ready):
            legal = np.flatnonzero(obs["action_mask"][slot])
            actions[slot] = 0 if 0 in legal else legal[0]
        _, dones, info = env.step(actions, active=active)
        assert not info["rejected"].any()
        for slot in np.flatnonzero(dones):
            lane, role = divmod(int(slot), 7)
            assert placements[lane, role] == 0
            placements[lane, role] = info["placement"][slot]
        for lane in np.flatnonzero(active & (placements > 0).all(axis=1)):
            assert len(set(placements[lane])) == 7
            assert set(placements[lane]) <= set(range(1, 9))
            assert env.episode_index(int(lane)) == 1
            active[lane] = False
        if not active.any():
            break
    assert not active.any()
    assert all(len(set(initial_seats[i:i + 7])) == 7 for i in (0, 7))
    # Inactive lanes stay at index 1: the auto-created next game is never stepped.
    env.step(np.zeros(env.n_slots, dtype=int), active=active)
    assert [env.episode_index(i) for i in range(2)] == [1, 1]


def test_ranking_scheduler_records_original_rotating_seats(tmp_path, monkeypatch):
    class Policy:
        def predict(self, obs, deterministic):
            assert not deterministic
            return np.array([0 if row[0] else np.flatnonzero(row)[0]
                             for row in obs["action_mask"]]), None

    spec = {"model_version": "v1", "model": "fake", "sha256": "fakehash",
            "count": 6, "name": "fake", "reward_shaping": None}
    checkpoint = {"sha256": "fakehash", "reward_shaping": None}
    monkeypatch.setattr(ranking, "digest", lambda _: "fakehash")
    monkeypatch.setattr(ranking, "inspect_checkpoint", lambda _: checkpoint)
    monkeypatch.setattr(ranking, "load_frozen_policy", lambda *a, **k: Policy())
    initial = tg.BatchEnv(64, seed=60000007, controlled=range(7), threads=1,
                          config={"turn_action_caps": [60]}, rotate_seats=True).slot_seats
    result = ranking.parallel_evaluate(tmp_path, tmp_path,
        {"protocol": {"games": 64, "seed": 60000007, "reference_pool": [spec]},
         "candidates": [spec]}, {}, "cpu")
    import json
    games = [json.loads(line) for line in (tmp_path / "ranking-games.jsonl").read_text().splitlines()]
    assert len(games) == 64
    for game in games:
        lane = game["lane"]
        assert [p["seat"] for p in game["players"]] == initial[lane * 7:lane * 7 + 7].tolist()
        assert sorted([p["placement"] for p in game["players"]] + game["v0_placements"]) == list(range(1, 9))
    assert len(result["v1"]["placements"]) == 64
