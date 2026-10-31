"""Batched environment: many eight-player games stepped together.

Each game is a full lobby. You control the seats in `controlled` (e.g. `(0,)` for one
learner against seven scripted bots, or `range(8)` for self-play); scripted bots play the
others. A *slot* is one controlled seat of one game: slot `i` is seat `controlled[i % k]` of
game `i // k`, with `k = len(controlled)`.

    env = BatchEnv(n_games=64, seed=0, controlled=(0,), threads=8)
    obs = env.observe()
    while training:
        actions = policy(obs["features"], obs["cards"], obs["action_mask"])  # int per slot
        rewards, dones, info = env.step(actions)
        obs = env.observe()

`observe()["ready"]` says which slots must act now; actions of the other slots are ignored.
Several seats of one game can be ready at once (the recruit phase is simultaneous). The
reward is the placement reward (1st = 1, last = -1) when a seat's episode ends, plus
`illegal_penalty` for a rejected action. A game whose controlled seats have all finished
starts again with a new seed. Results do not depend on `threads`.
"""

import json

import numpy as np

from taverngym._core import N_ACTIONS, OBS_CARDS, OBS_FLOATS
from taverngym._core import BatchEnv as _BatchEnv
from taverngym._core import Game


class BatchEnv:
    def __init__(self, n_games, seed=0, controlled=(0,), config=None, threads=1, rotate_seats=False):
        config_json = json.dumps(config) if config is not None else None
        self._env = _BatchEnv(int(n_games), int(seed), [int(p) for p in controlled],
                              config_json, int(threads), bool(rotate_seats))

    @property
    def n_games(self):
        return self._env.n_games

    @property
    def n_slots(self):
        return self._env.n_slots

    @property
    def controlled(self):
        return tuple(self._env.controlled)

    @property
    def slot_seats(self):
        """Actual seat of every slot, including per-game role rotation."""
        return np.asarray(self._env.slot_seats, dtype=np.int64)

    def reset_finished_first(self):
        """Reset games whose first role finished; keep its already-reported result."""
        self._env.reset_finished_first()

    @property
    def illegal_penalty(self):
        return self._env.illegal_penalty

    @illegal_penalty.setter
    def illegal_penalty(self, value):
        self._env.illegal_penalty = float(value)

    def slot(self, i):
        """(game index, seat) of slot `i`."""
        k = len(self.controlled)
        return i // k, int(self.slot_seats[i])

    def observe(self):
        """Per slot: `features` float32 [n_slots, OBS_FLOATS], `cards` int32
        [n_slots, OBS_CARDS] (card embedding indices, 0 = empty), `action_mask` bool
        [n_slots, N_ACTIONS] and `ready` bool [n_slots]."""
        f, c, m, r = self._env.observe()
        n = self.n_slots
        return {
            "features": np.frombuffer(f, dtype="<f4").reshape(n, OBS_FLOATS).copy(),
            "cards": np.frombuffer(c, dtype="<i4").reshape(n, OBS_CARDS).copy(),
            "action_mask": np.frombuffer(m, dtype=np.uint8).reshape(n, N_ACTIONS).astype(bool),
            "ready": np.frombuffer(r, dtype=np.uint8).astype(bool),
        }

    def step(self, actions, active=None):
        """Apply one action per slot. Returns (rewards float32 [n_slots], dones bool
        [n_slots], info) with `info["placement"]` (uint8, the final placement where done;
        0 if a simulator guard stopped the game) and `info["rejected"]` (bool)."""
        a = np.ascontiguousarray(np.asarray(actions).reshape(-1), dtype="<i8")
        if a.shape[0] != self.n_slots:
            raise ValueError(f"{a.shape[0]} actions for {self.n_slots} slots")
        flags = None
        if active is not None:
            flags = np.asarray(active, dtype=np.uint8)
            if flags.shape != (self.n_games,):
                raise ValueError("active must contain one flag per game")
            flags = flags.tobytes()
        rw, dn, pl, rj = self._env.step(a.tobytes(), flags)
        return (
            np.frombuffer(rw, dtype="<f4").copy(),
            np.frombuffer(dn, dtype=np.uint8).astype(bool),
            {
                "placement": np.frombuffer(pl, dtype=np.uint8).copy(),
                "rejected": np.frombuffer(rj, dtype=np.uint8).astype(bool),
            },
        )

    def save(self, i):
        """Game `i` and its episode state as JSON; `load(i, s)` restores it exactly."""
        return self._env.save(int(i))

    def load(self, i, state):
        self._env.load(int(i), state)

    def episode_index(self, i):
        return self._env.episode_index(int(i))

    def game(self, i) -> Game:
        """An independent copy of game `i` (inspect it with `view`, `state_json`...)."""
        return self._env.game(int(i))
