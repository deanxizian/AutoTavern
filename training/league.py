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
from .common import atomic_json, tensors
from .decoding import CancelRetryGuard
from .diagnostics import check_game_guards
from .turns import DRIVER, next_ready, splitmix64
from .attachments import AttachmentRows, policy_kwargs


def observation_rows(games, rows, *, public_attachments=False):
    rows = list(rows)
    features, cards, masks = [], [], []
    for lane, seat in rows:
        f, c = games[lane].observe(seat)
        features.append(np.frombuffer(f, dtype="<f4"))
        cards.append(np.frombuffer(c, dtype="<i4"))
        masks.append(np.frombuffer(games[lane].legal_mask(seat), np.uint8).astype(bool))
    result = {"features": np.stack(features), "cards": np.stack(cards),
              "action_mask": np.stack(masks), "ready": np.ones(len(rows), bool)}
    if public_attachments:
        result["attachments"] = AttachmentRows.from_games(games, rows)
    return result


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
                 config=None, threads=1, inference_batch=256, cancel_guard=False,
                 training_turn_limit=0, truncation_dir=None, opponent_graph_max_batch=0,
                 public_attachments=False):
        if type(public_attachments) is not bool:
            raise ValueError("Public attachment observations must be explicitly enabled or disabled")
        if public_attachments and not hasattr(tg.Game, "attachment_observation"):
            raise ValueError("The native build has no public attachment observation API")
        self.public_attachments = public_attachments
        if n_games < 1 or not opponents or len(opponents) != len(opponent_ids):
            raise ValueError("Supply positive lanes and a nonempty, identified opponent pool")
        if inference_batch < 1 or not 0 <= seed < 2**64:
            raise ValueError("Invalid inference batch or seed")
        if type(training_turn_limit) is not int or training_turn_limit < 0:
            raise ValueError("The training turn limit must be a nonnegative integer")
        if training_turn_limit and truncation_dir is None:
            raise ValueError("Time-limited training must retain truncation evidence")
        if type(opponent_graph_max_batch) is not int or not 0 <= opponent_graph_max_batch <= 64:
            raise ValueError("Opponent graph batch limit must be an integer in 0..64")
        if opponent_graph_max_batch and torch.device(device).type != "cuda":
            raise ValueError("Opponent graphs require a CUDA device")
        self.n_games = self.n_slots = n_games
        self.seed, self.config = seed, config or {}
        self.training_turn_limit = training_turn_limit
        self.truncation_dir = Path(truncation_dir) if truncation_dir is not None else None
        self.truncated_episodes = [0] * n_games
        self.opponents, self.opponent_ids = list(opponents), list(opponent_ids)
        if opponent_graph_max_batch:
            from .inference import FrozenOpponentGraphs
            cache = {}
            for model in opponents:
                if model is not None and id(model) not in cache:
                    cache[id(model)] = (model if getattr(model, "requires_attachments", False)
                                        else FrozenOpponentGraphs(model, opponent_graph_max_batch))
            self.opponents = [cache[id(model)] if model is not None else None for model in opponents]
        self.device, self.inference_batch = torch.device(device), inference_batch
        self.threads = min(n_games, max(1, threads))
        self.executor = ThreadPoolExecutor(self.threads) if self.threads > 1 else None
        self.decoder = CancelRetryGuard() if cancel_guard else None
        self.engine_hash = tg.build_info()["source_hash"]
        self.indices = [0] * n_games
        self.learners = [0] * n_games
        self.assignments = [None] * n_games
        self.opponent_decisions = [0] * n_games
        self.cursors = [0] * n_games
        self.games = [None] * n_games
        for model in self.opponents:
            if model is not None:
                model.eval()
        for lane in range(n_games):
            self._fresh(lane)
        if training_turn_limit:
            native_limit = json.loads(self.games[0].state_json())["cfg"]["max_turns"]
            if native_limit and training_turn_limit > native_limit:
                self.close()
                raise ValueError("Training turn limit exceeds the native safety limit")
        self._drive(list(range(n_games)))

    def _fresh(self, lane):
        index = self.indices[lane]
        game_seed = splitmix64(splitmix64(self.seed ^ splitmix64(lane)) ^ index)
        self.games[lane] = tg.Game(game_seed, json.dumps(self.config))
        self.cursors[lane] = 0
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
        if any(self.cursors[lane] != seat or not self.games[lane].needs_action(seat)
               for lane, seat in rows):
            raise RuntimeError("A waiting learner escaped the league transition driver")
        obs = observation_rows(self.games, rows, public_attachments=self.public_attachments)
        return (self.decoder.apply(obs, [lane * 8 + seat for lane, seat in rows])
                if self.decoder is not None else obs)

    def _apply_chunk(self, chunk):
        for lane, seat, action in chunk:
            game = self.games[lane]
            if not game.legal_mask(seat)[int(action)]:
                raise ValueError("League action is masked at its decision point")
            game.step(seat, int(action))
            is_learner = seat == self.learners[lane]
            if self.decoder is not None and (is_learner or
                    self.opponents[self.assignments[lane][seat]] is not None):
                self.decoder.record([action], [lane * 8 + seat])
            self.cursors[lane] = (seat + 1) % 8
            if not is_learner:
                self.opponent_decisions[lane] += 1

    def _apply(self, work):
        if self.executor is None or len(work) < 2:
            self._apply_chunk(work)
        else:
            size = math.ceil(len(work) / self.threads)
            list(self.executor.map(self._apply_chunk,
                                   [work[i:i + size] for i in range(0, len(work), size)]))

    @torch.inference_mode()
    def _tick(self, acting_rows):
        # Each lane contributes at most one player. No earlier action in the same
        # lobby can invalidate a Discover, hand slot or target before execution.
        actions, groups = {}, {}
        for lane, seat in acting_rows:
            model = self.opponents[self.assignments[lane][seat]]
            if model is None:
                actions[lane] = self.games[lane].bot_action(seat)
            else:
                groups.setdefault(id(model), (model, []))[1].append((lane, seat))
        for model, rows in groups.values():
            for start in range(0, len(rows), self.inference_batch):
                batch = rows[start:start + self.inference_batch]
                obs = observation_rows(self.games, batch,
                                       public_attachments=getattr(model, "requires_attachments", False))
                identities = [lane * 8 + seat for lane, seat in batch]
                if self.decoder is not None:
                    obs = self.decoder.apply(obs, identities)
                logits, value = model(*tensors(obs, self.device), **policy_kwargs(model, obs, self.device))
                if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                    raise RuntimeError("Frozen opponent produced non-finite outputs")
                selected = logits.argmax(-1).cpu().numpy()
                for (lane, seat), action in zip(batch, selected):
                    actions[lane] = action
        self._apply([(lane, seat, actions[lane]) for lane, seat in acting_rows])

    def _drive(self, lanes):
        for _ in range(30_000):
            rows = []
            for lane in lanes:
                game, learner = self.games[lane], self.learners[lane]
                if game.placements()[learner] or game.is_over():
                    continue
                seat = next_ready(game, self.cursors[lane])
                self.cursors[lane] = seat
                if seat != learner:
                    rows.append((lane, seat))
            if not rows:
                return
            self._tick(rows)
        raise RuntimeError("League transition exceeded the driver limit")

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if len(actions) != self.n_games:
            raise ValueError("One action is required per learning lane")
        obs = self.observe()
        if np.any(actions < 0) or np.any(actions >= tg.N_ACTIONS):
            raise ValueError("Learner action is out of range")
        if not obs["action_mask"][np.arange(self.n_games), actions].all():
            raise ValueError("Learner action is masked")
        self._apply([(lane, self.learners[lane], action) for lane, action in enumerate(actions)])
        self._drive(list(range(self.n_games)))
        rewards = np.zeros(self.n_games, np.float32)
        done = np.zeros(self.n_games, bool)
        truncated = np.zeros(self.n_games, bool)
        placement = np.zeros(self.n_games, np.uint8)
        reset = []
        for lane in range(self.n_games):
            game, learner = self.games[lane], self.learners[lane]
            place = game.placements()[learner]
            if place:
                check_game_guards(game, context={"lane": lane, "episode": self.indices[lane]})
                if not 1 <= place <= 8:
                    raise RuntimeError("Invalid league placement")
                rewards[lane] = np.float32(1) - np.float32(2) * np.float32(place - 1) / np.float32(7)
                done[lane], placement[lane] = True, place
                reset.append(lane)
            elif game.is_over():
                raise RuntimeError("League game ended without the learner's placement")
            elif self.training_turn_limit and game.turn >= self.training_turn_limit:
                # Stop at an actual ready-player observation BEFORE the native end-of-
                # round guard invents placements by Health. This is a training horizon,
                # not a game outcome; its value bootstrap is handled by the PPO caller.
                check_game_guards(game, context={"lane": lane, "episode": self.indices[lane]})
                evidence = {"reason": "training_turn_limit", "turn": game.turn,
                            "environment": json.loads(self.save(lane))}
                self.truncation_dir.mkdir(parents=True, exist_ok=True)
                path = self.truncation_dir / f"lane{lane:05d}-episode{self.indices[lane]:08d}.json"
                if path.exists():
                    if json.loads(path.read_text()) != evidence:
                        raise RuntimeError("Truncation evidence would overwrite a different state")
                else:
                    atomic_json(path, evidence)
                self.truncated_episodes[lane] += 1
                done[lane] = truncated[lane] = True
                reset.append(lane)
        info = {"placement": placement, "rejected": np.zeros(self.n_games, bool),
                "truncated": truncated}
        lanes = np.flatnonzero(truncated)
        if len(lanes):
            info["truncated_lanes"] = lanes
            rows = [(int(lane), self.learners[lane]) for lane in lanes]
            info["final_observation"] = observation_rows(self.games, rows,
                                                       public_attachments=self.public_attachments)
            if self.decoder is not None:
                info["final_observation"] = self.decoder.apply(
                    info["final_observation"], [lane * 8 + seat for lane, seat in rows])
        for lane in reset:
            self.indices[lane] += 1
            self._fresh(lane)
        self._drive(reset)
        return rewards, done, info

    def game(self, lane):
        return self.games[lane].copy()

    def episode_index(self, lane):
        return self.indices[lane]

    def save(self, lane):
        data = {
            "format": "league-v2", "driver": DRIVER, "cursor": self.cursors[lane],
            "engine_hash": self.engine_hash,
            "seed": self.seed, "n_games": self.n_games, "lane": lane,
            "config": self.config, "opponent_ids": self.opponent_ids,
            "training_turn_limit": self.training_turn_limit,
            "truncated_episodes": self.truncated_episodes[lane],
            "episode": self.indices[lane], "learner": self.learners[lane],
            "assignments": self.assignments[lane],
            "opponent_decisions": self.opponent_decisions[lane],
            "game": self.games[lane].state_json(),
            "decoder": ([self.decoder.save(lane * 8 + seat) for seat in range(8)]
                        if self.decoder is not None else None),
        }
        if self.public_attachments:
            data["attachment_version"] = 1
        return json.dumps(data, sort_keys=True)

    def load(self, lane, state):
        data = json.loads(state)
        if data.get("attachment_version", 0) != int(self.public_attachments):
            raise ValueError("League resume changes the public attachment observation version")
        required = {"format": "league-v2", "driver": DRIVER, "engine_hash": self.engine_hash,
                    "seed": self.seed, "n_games": self.n_games, "lane": lane,
                    "config": self.config, "opponent_ids": self.opponent_ids}
        for key, value in required.items():
            if data[key] != value:
                raise ValueError(f"League resume changes {key}")
        if data.get("training_turn_limit", 0) != self.training_turn_limit:
            raise ValueError("League resume changes training_turn_limit; start a new stage")
        if (data["decoder"] is not None) != (self.decoder is not None):
            raise ValueError("League resume changes cancellation decoding")
        self.indices[lane], self.learners[lane] = data["episode"], data["learner"]
        self.cursors[lane] = data["cursor"]
        self.assignments[lane] = data["assignments"]
        self.opponent_decisions[lane] = data["opponent_decisions"]
        self.truncated_episodes[lane] = data.get("truncated_episodes", 0)
        self.games[lane] = tg.Game.from_json(data["game"])
        if self.decoder is not None:
            for seat, decoder in enumerate(data["decoder"]):
                self.decoder.load(lane * 8 + seat, decoder)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def inference_stats(self):
        models = {id(model): model for model in self.opponents if model is not None}
        return [model.inference_stats() for model in models.values()
                if hasattr(model, "inference_stats")]
