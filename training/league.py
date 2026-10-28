"""One PPO learner per lobby against a fixed pool of player-view policies.

Independent Game lanes let a ready learner wait unchanged while other lanes finish
their opponents' recruit actions. No dummy learner actions or waiting-state GAE rows
are introduced. Each step ends at that learner's next decision or terminal placement.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import numpy as np
import torch

import taverngym as tg
from .common import tensors
from .decoding import CancelRetryGuard
from .diagnostics import check_game_guards


def splitmix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def observation_rows(games, rows):
    features, cards, masks = [], [], []
    for lane, seat in rows:
        f, c = games[lane].observe(seat)
        features.append(np.frombuffer(f, dtype="<f4"))
        cards.append(np.frombuffer(c, dtype="<i4"))
        masks.append(np.frombuffer(games[lane].legal_mask(seat), np.uint8).astype(bool))
    return {"features": np.stack(features), "cards": np.stack(cards),
            "action_mask": np.stack(masks), "ready": np.ones(len(rows), bool)}


def load_opponents(specifications, device):
    """Repeated entries weight the fixed pool; identical paths share one frozen model."""
    from .arena import frozen_checkpoint
    cache, models, metadata = {}, [], []
    for specification in specifications:
        key = "bot" if specification == "bot" else str(Path(specification).resolve())
        if key not in cache:
            cache[key] = ((None, {"kind": "bundled_heuristic", "sha256": "bot"})
                          if key == "bot" else frozen_checkpoint(Path(key), device))
            if cache[key][0] is not None:
                cache[key][0].eval().requires_grad_(False)
        model, meta = cache[key]
        models.append(model)
        metadata.append(meta)
    return models, metadata


class LeagueEnv:
    def __init__(self, n_games, *, seed, opponents, opponent_ids, device,
                 config=None, threads=1, inference_batch=256, cancel_guard=False):
        if n_games < 1 or not opponents or len(opponents) != len(opponent_ids):
            raise ValueError("Supply positive lanes and a nonempty, identified opponent pool")
        if inference_batch < 1 or not 0 <= seed < 2**64:
            raise ValueError("Invalid inference batch or seed")
        self.n_games = self.n_slots = n_games
        self.seed, self.config = seed, config or {}
        self.opponents, self.opponent_ids = list(opponents), list(opponent_ids)
        self.device, self.inference_batch = torch.device(device), inference_batch
        self.threads = min(n_games, max(1, threads))
        self.executor = ThreadPoolExecutor(self.threads) if self.threads > 1 else None
        self.decoder = CancelRetryGuard() if cancel_guard else None
        self.engine_hash = tg.build_info()["source_hash"]
        self.indices = [0] * n_games
        self.learners = [0] * n_games
        self.assignments = [None] * n_games
        self.opponent_decisions = [0] * n_games
        self.games = [None] * n_games
        for model in opponents:
            if model is not None:
                model.eval()
        for lane in range(n_games):
            self._fresh(lane)

    def _fresh(self, lane):
        index = self.indices[lane]
        game_seed = splitmix64(splitmix64(self.seed ^ splitmix64(lane)) ^ index)
        self.games[lane] = tg.Game(game_seed, json.dumps(self.config))
        if self.games[lane].num_players != 8:
            raise ValueError("The league requires eight-player lobbies")
        learner = self.learners[lane] = (lane + index) % 8
        offset = splitmix64(game_seed) % len(self.opponents)
        self.assignments[lane] = [
            -1 if seat == learner else (offset + (seat - learner - 1) % 8) % len(self.opponents)
            for seat in range(8)]
        if self.decoder is not None:
            for seat in range(8):
                self.decoder.reset(lane * 8 + seat)

    def observe(self):
        rows = [(lane, self.learners[lane]) for lane in range(self.n_games)]
        if any(not self.games[lane].needs_action(seat) for lane, seat in rows):
            raise RuntimeError("A waiting learner escaped the league transition driver")
        obs = observation_rows(self.games, rows)
        return (self.decoder.apply(obs, [lane * 8 + seat for lane, seat in rows])
                if self.decoder is not None else obs)

    def _apply_chunk(self, chunk):
        for lane, ready, actions in chunk:
            game = self.games[lane]
            # Match native BatchEnv: snapshot readiness before the first action.
            for seat in range(8):
                if ready[seat] and not game.is_over() and game.needs_action(seat):
                    game.step(seat, int(actions[seat]))

    @torch.inference_mode()
    def _tick(self, lanes, learner_actions=None):
        actions, readiness, groups = {}, {}, {}
        for lane in lanes:
            game = self.games[lane]
            ready = readiness[lane] = [game.needs_action(seat) for seat in range(8)]
            if not any(ready):
                check_game_guards(game, context={"lane": lane, "episode": self.indices[lane]})
                raise RuntimeError("League game has no ready seat and no terminal learner")
            actions[lane] = np.zeros(8, np.int64)
            learner = self.learners[lane]
            if ready[learner]:
                if learner_actions is None:
                    raise RuntimeError("A ready learner must not advance during other lanes' waits")
                actions[lane][learner] = learner_actions[lane]
            for seat in range(8):
                if seat == learner or not ready[seat]:
                    continue
                model = self.opponents[self.assignments[lane][seat]]
                self.opponent_decisions[lane] += 1
                if model is None:
                    actions[lane][seat] = game.bot_action(seat)
                else:
                    groups.setdefault(id(model), (model, []))[1].append((lane, seat))
        for model, rows in groups.values():
            for start in range(0, len(rows), self.inference_batch):
                batch = rows[start:start + self.inference_batch]
                obs = observation_rows(self.games, batch)
                identities = [lane * 8 + seat for lane, seat in batch]
                if self.decoder is not None:
                    obs = self.decoder.apply(obs, identities)
                logits, value = model(*tensors(obs, self.device))
                if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                    raise RuntimeError("Frozen opponent produced non-finite outputs")
                selected = logits.argmax(-1).cpu().numpy()
                if self.decoder is not None:
                    self.decoder.record(selected, identities)
                for (lane, seat), action in zip(batch, selected):
                    actions[lane][seat] = action
        work = [(lane, readiness[lane], actions[lane]) for lane in lanes]
        if self.executor is None or len(work) < 2:
            self._apply_chunk(work)
        else:
            size = math.ceil(len(work) / self.threads)
            list(self.executor.map(self._apply_chunk,
                                   [work[i:i + size] for i in range(0, len(work), size)]))

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if len(actions) != self.n_games:
            raise ValueError("One action is required per learning lane")
        obs = self.observe()
        if np.any(actions < 0) or np.any(actions >= tg.N_ACTIONS):
            raise ValueError("Learner action is out of range")
        if not obs["action_mask"][np.arange(self.n_games), actions].all():
            raise ValueError("Learner action is masked")
        if self.decoder is not None:
            self.decoder.record(actions, [i * 8 + self.learners[i] for i in range(self.n_games)])
        self._tick(list(range(self.n_games)), actions)
        rewards = np.zeros(self.n_games, np.float32)
        done = np.zeros(self.n_games, bool)
        placement = np.zeros(self.n_games, np.uint8)
        pending = list(range(self.n_games))
        for _ in range(30_000):
            waiting = []
            for lane in pending:
                game, learner = self.games[lane], self.learners[lane]
                place = game.placements()[learner]
                if place:
                    check_game_guards(game, context={"lane": lane, "episode": self.indices[lane]})
                    if not 1 <= place <= 8:
                        raise RuntimeError("Invalid league placement")
                    rewards[lane] = np.float32(1) - np.float32(2) * np.float32(place - 1) / np.float32(7)
                    done[lane], placement[lane] = True, place
                    self.indices[lane] += 1
                    self._fresh(lane)
                elif game.is_over():
                    raise RuntimeError("League game ended without the learner's placement")
                elif not game.needs_action(learner):
                    waiting.append(lane)
            if not waiting:
                return rewards, done, {"placement": placement,
                                       "rejected": np.zeros(self.n_games, bool)}
            # Completed or ready lanes remain untouched while these lanes catch up.
            self._tick(waiting)
            pending = waiting
        raise RuntimeError("League transition exceeded the driver limit")

    def game(self, lane):
        return self.games[lane].copy()

    def episode_index(self, lane):
        return self.indices[lane]

    def save(self, lane):
        return json.dumps({
            "format": "league-v1", "engine_hash": self.engine_hash,
            "seed": self.seed, "n_games": self.n_games, "lane": lane,
            "config": self.config, "opponent_ids": self.opponent_ids,
            "episode": self.indices[lane], "learner": self.learners[lane],
            "assignments": self.assignments[lane],
            "opponent_decisions": self.opponent_decisions[lane],
            "game": self.games[lane].state_json(),
            "decoder": ([self.decoder.save(lane * 8 + seat) for seat in range(8)]
                        if self.decoder is not None else None),
        }, sort_keys=True)

    def load(self, lane, state):
        data = json.loads(state)
        required = {"format": "league-v1", "engine_hash": self.engine_hash,
                    "seed": self.seed, "n_games": self.n_games, "lane": lane,
                    "config": self.config, "opponent_ids": self.opponent_ids}
        for key, value in required.items():
            if data[key] != value:
                raise ValueError(f"League resume changes {key}")
        if (data["decoder"] is not None) != (self.decoder is not None):
            raise ValueError("League resume changes cancellation decoding")
        self.indices[lane], self.learners[lane] = data["episode"], data["learner"]
        self.assignments[lane] = data["assignments"]
        self.opponent_decisions[lane] = data["opponent_decisions"]
        self.games[lane] = tg.Game.from_json(data["game"])
        if self.decoder is not None:
            for seat, decoder in enumerate(data["decoder"]):
                self.decoder.load(lane * 8 + seat, decoder)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
