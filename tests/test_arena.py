"""Arena regressions: seat bias, auto-reset leakage and waiting-seat inference."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from training.arena import play_arena, rotations, wilson


class EndTurn(torch.nn.Module):
    def forward(self, features, cards, masks):
        assert masks.any(axis=1).all(), "waiting seats reached inference"
        logits = torch.zeros_like(masks, dtype=torch.float32).masked_fill(~masks, -1e9)
        return logits, torch.zeros(len(masks))


class UnequalDurationEnv:
    """Lane zero resets before lane seven ends; reset results must be ignored."""
    def __init__(self, n_games, **kwargs):
        self.n_slots = n_games * 8
        self.n_games = n_games
        self.tick = 0

    def observe(self):
        ready = np.tile(np.arange(8) % 2 == self.tick % 2, self.n_games)
        return {"ready": ready, "features": np.zeros((self.n_slots, 1), np.float32),
                "cards": np.zeros((self.n_slots, 1), np.int32),
                "action_mask": ready[:, None]}

    def step(self, actions):
        self.tick += 1
        done = np.repeat(self.tick % (np.arange(self.n_games) + 1) == 0, 8)
        # Different placements in subsequent episodes reveal accidental overwrites.
        places = np.concatenate([np.roll(np.arange(1, 9), self.tick // (lane + 1) - 1)
                                 for lane in range(self.n_games)]).astype(np.uint8)
        return np.zeros(self.n_slots), done, {"placement": places,
                                             "rejected": np.zeros(self.n_slots, bool)}


def test_every_role_occupies_every_seat_equally():
    roles = rotations(24)
    for role in range(8):
        np.testing.assert_array_equal((roles == role).sum(axis=0), np.full(8, 3))
    with pytest.raises(ValueError):
        rotations(9)


def test_first_lobby_only_and_waiting_slots_never_reach_policy():
    model = EndTurn().train()
    result = play_arena([model] * 8, games=8, seed=17, threads=1, config={},
                        device=torch.device("cpu"), env_factory=UnequalDurationEnv,
                        audit_guards=False)
    assert model.training, "evaluation must restore the caller's training mode"
    assert result["driver_steps"] == 8
    assert result["candidate"]["placement_counts"] == [1] * 8
    assert result["candidate"]["mean_placement"] == 4.5
    assert all(row["placement_by_seat"] == list(range(1, 9)) for row in result["lobbies"])


def test_real_arena_finishes_whole_lobbies_reproducibly():
    kwargs = dict(games=8, seed=42, config={"turn_action_caps": [32]},
                  device=torch.device("cpu"))
    a = play_arena([None] * 8, threads=1, **kwargs)
    b = play_arena([None] * 8, threads=2, **kwargs)
    assert a["lobbies"] == b["lobbies"]
    assert a["candidate"]["games"] == 8
    for row in a["lobbies"]:
        assert sorted(row["placement_by_seat"]) == list(range(1, 9))


def test_rate_intervals_remain_conservative_at_endpoints():
    assert wilson(0, 100)[0] == 0
    assert 0 < wilson(0, 100)[1] < .05
    assert .95 < wilson(100, 100)[0] < 1
    assert wilson(100, 100)[1] == 1
