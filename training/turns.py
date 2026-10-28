"""Seat-ordered decisions with no stale actions inside an individual lobby."""

from concurrent.futures import ThreadPoolExecutor
import json
import math

import numpy as np

import taverngym as tg
from .diagnostics import check_game_guards

DRIVER = "round_robin_player_views_v1"


def splitmix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def next_ready(game, cursor):
    for offset in range(game.num_players):
        seat = (cursor + offset) % game.num_players
        if game.needs_action(seat):
            return seat
    if not game.is_over():
        check_game_guards(game, context={"driver": DRIVER})
        raise RuntimeError("A nonterminal game has no ready seat")
    return None


class RoundRobinArena:
    """One decision per live lobby per step; retain terminal games for auditing."""
    terminal_is_retained = True
    driver = DRIVER

    def __init__(self, n_games, *, seed, controlled, threads, config):
        if tuple(controlled) != tuple(range(8)):
            raise ValueError("Arena controls all eight players")
        self.n_games, self.n_slots = n_games, n_games * 8
        self.games = [tg.Game(splitmix64(splitmix64(seed ^ splitmix64(i))), json.dumps(config))
                      for i in range(n_games)]
        self.cursors = [0] * n_games
        self.finished = np.zeros((n_games, 8), bool)
        self.threads = min(n_games, max(1, threads))
        self.executor = ThreadPoolExecutor(self.threads) if self.threads > 1 else None

    def game(self, lane):
        return self.games[lane].copy()

    def observe(self):
        obs = {"features": np.zeros((self.n_slots, tg.OBS_FLOATS), np.float32),
               "cards": np.zeros((self.n_slots, tg.OBS_CARDS), np.int32),
               "action_mask": np.zeros((self.n_slots, tg.N_ACTIONS), bool),
               "ready": np.zeros(self.n_slots, bool)}
        for lane, game in enumerate(self.games):
            seat = next_ready(game, self.cursors[lane])
            if seat is None:
                continue
            self.cursors[lane] = seat
            slot = lane * 8 + seat
            features, cards = game.observe(seat)
            obs["features"][slot] = np.frombuffer(features, "<f4")
            obs["cards"][slot] = np.frombuffer(cards, "<i4")
            obs["action_mask"][slot] = np.frombuffer(game.legal_mask(seat), np.uint8)
            obs["ready"][slot] = True
        return obs

    def _apply(self, row):
        lane, seat, action = row
        game = self.games[lane]
        if not 0 <= action < tg.N_ACTIONS or not game.legal_mask(seat)[action]:
            raise ValueError("Arena action is masked at its decision point")
        game.step(seat, action)
        self.cursors[lane] = (seat + 1) % 8

    def _apply_chunk(self, rows):
        for row in rows:
            self._apply(row)

    def step(self, actions):
        actions = np.asarray(actions).reshape(-1)
        if len(actions) != self.n_slots:
            raise ValueError("Arena needs one action slot per player")
        work = []
        for lane, game in enumerate(self.games):
            seat = next_ready(game, self.cursors[lane])
            if seat is not None:
                work.append((lane, seat, int(actions[lane * 8 + seat])))
        if self.executor is None or len(work) < 2:
            self._apply_chunk(work)
        else:
            size = math.ceil(len(work) / self.threads)
            list(self.executor.map(self._apply_chunk,
                                   [work[i:i + size] for i in range(0, len(work), size)]))
        placements = np.asarray([list(g.placements()) for g in self.games], np.uint8)
        done = (placements != 0) & ~self.finished
        self.finished |= done
        rewards = np.where(done, 1. - 2. * (placements.astype(np.float32) - 1.) / 7., 0.)
        return rewards.reshape(-1), done.reshape(-1), {
            "placement": placements.reshape(-1), "rejected": np.zeros(self.n_slots, bool)}

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
