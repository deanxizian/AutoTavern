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
        self._restore_versions = np.zeros(int(n_games), dtype=np.int64)
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

    def step(self, actions, active=None, forfeit=None, skipped=None):
        """Apply one action per slot. Returns (rewards float32 [n_slots], dones bool
        [n_slots], info) with `info["placement"]` (uint8, the final placement where done;
        0 if a simulator guard stopped the game), `info["rejected"]` (bool), and
        `info["deferred"]` for an unexecuted action invalidated by an earlier seat.
        Observe again before choosing a replacement for a deferred action."""
        a = np.ascontiguousarray(np.asarray(actions).reshape(-1), dtype="<i8")
        if a.shape[0] != self.n_slots:
            raise ValueError(f"{a.shape[0]} actions for {self.n_slots} slots")
        selected = None if forfeit is None else np.asarray(forfeit, dtype=bool)
        if selected is not None and selected.shape != (self.n_slots,):
            raise ValueError("forfeit requires one flag per slot")
        if selected is not None and selected.any():
            active_games = (np.ones(self.n_games, dtype=bool) if active is None
                            else np.asarray(active, dtype=bool).copy())
            if active_games.shape != (self.n_games,):
                raise ValueError("active must contain one flag per game")
            forfeiting_games = selected.reshape(self.n_games, -1).any(axis=1)
            if (forfeiting_games & ~active_games).any():
                raise ValueError("Cannot forfeit an inactive game")
            terminal_rw, terminal_dn, terminal_info = self.forfeit(selected)
            active_games &= ~forfeiting_games
            rw, dn, info = self.step(actions, active=active_games)
            rw += terminal_rw
            dn |= terminal_dn
            info["placement"][terminal_dn] = terminal_info["placement"][terminal_dn]
            info["forfeited"] = selected.copy()
            return rw, dn, info
        flags = None
        if active is not None:
            flags = np.asarray(active, dtype=np.uint8)
            if flags.shape != (self.n_games,):
                raise ValueError("active must contain one flag per game")
            flags = flags.tobytes()
        held = None
        if skipped is not None:
            if forfeit is not None:
                raise ValueError("Use either role skipping or direct forfeits")
            held_array = np.asarray(skipped, dtype=np.uint8)
            if held_array.shape != (self.n_slots,):
                raise ValueError("skipped requires one flag per slot")
            held = held_array.tobytes()
        rw, dn, pl, status = self._env.step(a.tobytes(), flags, held)
        return self._decode_result(rw, dn, pl, status)

    def forfeit(self, selected):
        """Surrender selected ready roles simultaneously, with native placement ties.

        Returns the same reward/done/placement structure as step. No action is
        taken in other games. Historical callers never invoke this opt-in API.
        """
        flags = np.asarray(selected, dtype=np.uint8)
        if flags.shape != (self.n_slots,):
            raise ValueError("forfeit requires one flag per slot")
        return self._decode_result(*self._env.forfeit(flags.tobytes()))

    @staticmethod
    def _decode_result(rw, dn, pl, status):
        status = np.frombuffer(status, dtype=np.uint8)
        return (
            np.frombuffer(rw, dtype="<f4").copy(),
            np.frombuffer(dn, dtype=np.uint8).astype(bool),
            {
                "placement": np.frombuffer(pl, dtype=np.uint8).copy(),
                "rejected": (status & 1).astype(bool),
                "deferred": (status & 2).astype(bool),
            },
        )

    def save(self, i):
        """Game `i` and its episode state as JSON; `load(i, s)` restores it exactly."""
        return self._env.save(int(i))

    def load(self, i, state):
        self._env.load(int(i), state)
        self._restore_versions[int(i)] += 1

    def episode_index(self, i):
        return self._env.episode_index(int(i))

    def game(self, i) -> Game:
        """An independent copy of game `i` (inspect it with `view`, `state_json`...)."""
        return self._env.game(int(i))
