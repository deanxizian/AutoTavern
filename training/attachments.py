"""Variable-length public attachment bags, aligned with legacy observation card slots.

Only the dedicated native accessor supplies these rows. Serialized games and debug
views are not observation adapters. Counts and IDs stay compact on CPU in a rollout;
only the selected minibatch is packed for the policy. Nothing is silently truncated.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import taverngym as tg


class AttachmentTensors(NamedTuple):
    cards: torch.Tensor
    offsets: torch.Tensor  # CSR: one bag per (batch row, legacy card slot), plus final end.


class AttachmentRows:
    version = 1

    def __init__(self, counts, cards):
        counts, cards = np.asarray(counts), np.asarray(cards)
        if (counts.ndim != 2 or counts.shape[1] != tg.OBS_CARDS
                or counts.dtype.kind not in "iu" or cards.ndim != 1
                or cards.dtype.kind not in "iu"):
            raise ValueError("Attachment rows need integer counts[rows, OBS_CARDS] and flat card IDs")
        if np.any(counts < 0) or np.any(counts > np.iinfo(np.int32).max):
            raise ValueError("Attachment count is outside the int32 observation contract")
        if np.any(cards <= 0) or np.any(cards >= tg.N_CARDS):
            raise ValueError("Attachment card ID is outside the card table")
        if int(counts.sum(dtype=np.int64)) != len(cards):
            raise ValueError("Attachment counts do not cover exactly the supplied card IDs")
        self.counts = np.array(counts, dtype=np.int32, copy=True)
        self.cards = np.array(cards, dtype=np.int32, copy=True)
        self._row_offsets = np.concatenate((np.zeros(1, np.int64),
            np.cumsum(self.counts.sum(axis=1, dtype=np.int64), dtype=np.int64)))

    def __len__(self):
        return len(self.counts)

    @property
    def nbytes(self):
        return self.counts.nbytes + self.cards.nbytes + self._row_offsets.nbytes

    @classmethod
    def empty(cls, rows):
        if type(rows) is not int or rows < 0:
            raise ValueError("Attachment row count must be a nonnegative integer")
        return cls(np.zeros((rows, tg.OBS_CARDS), np.int32), np.empty(0, np.int32))

    @classmethod
    def from_games(cls, games, rows):
        counts, cards = [], []
        for lane, seat in rows:
            raw_offsets, raw_cards = games[lane].attachment_observation(seat)
            offsets = np.frombuffer(raw_offsets, dtype="<i4")
            ids = np.frombuffer(raw_cards, dtype="<i4")
            if (offsets.shape != (tg.OBS_CARDS + 1,) or offsets[0] != 0
                    or offsets[-1] != len(ids) or np.any(offsets[1:] < offsets[:-1])):
                raise ValueError("Native attachment offsets do not match the observation contract")
            counts.append(np.diff(offsets))
            cards.append(ids)
        if not counts:
            return cls.empty(0)
        return cls(np.stack(counts), np.concatenate(cards))

    @classmethod
    def from_env(cls, env, rows):
        rows = list(rows)
        # Round-robin/league drivers retain games; generic batch drivers expose copies.
        # In both cases only the dedicated public accessor supplies policy information.
        games = getattr(env, "games", None)
        if games is None:
            games = {lane: env.game(lane) for lane, _ in rows}
        return cls.from_games(games, rows)

    @classmethod
    def concatenate(cls, batches):
        batches = list(batches)
        if not batches:
            return cls.empty(0)
        if any(not isinstance(batch, cls) for batch in batches):
            raise TypeError("Concatenate only attachment row batches")
        return cls(np.concatenate([b.counts for b in batches]),
                   np.concatenate([b.cards for b in batches]))

    def __getitem__(self, selection):
        # NumPy semantics preserve repeated/reordered rows and reject invalid index types.
        rows = np.asarray(np.arange(len(self))[selection], dtype=np.int64).reshape(-1)
        if not len(rows):
            return self.empty(0)
        ids = np.concatenate([self.cards[self._row_offsets[i]:self._row_offsets[i + 1]]
                              for i in rows])
        return type(self)(self.counts[rows], ids)

    def tensors(self, device):
        offsets = np.concatenate((np.zeros(1, np.int64),
            np.cumsum(self.counts.reshape(-1), dtype=np.int64)))
        return AttachmentTensors(torch.as_tensor(self.cards, dtype=torch.int64, device=device),
                                 torch.as_tensor(offsets, dtype=torch.int64, device=device))


def policy_kwargs(policy, observation, device):
    """Supply supplemental inputs only to policies whose architecture requires them."""
    result = {}
    if getattr(policy, "requires_attachments", False):
        rows = observation.get("attachments")
        if not isinstance(rows, AttachmentRows) or len(rows) != len(observation["features"]):
            raise ValueError("A public-attachment policy needs aligned attachment observation rows")
        result["attachments"] = rows.tensors(device)
    if getattr(policy, "requires_strategy", False):
        prior = observation.get("strategy_bias")
        if (not isinstance(prior, np.ndarray) or prior.dtype != np.float32
                or prior.shape != observation["action_mask"].shape
                or not np.isfinite(prior).all() or np.any(np.abs(prior) > 6)):
            raise ValueError("An expert policy needs aligned bounded strategy priors")
        result["strategy_bias"] = torch.as_tensor(prior, device=device)
    if getattr(policy, "requires_growth", False):
        from .growth import N_FEATURES
        growth = observation.get("growth")
        if (not isinstance(growth, np.ndarray) or growth.dtype != np.float32
                or growth.shape != (len(observation["features"]), N_FEATURES)
                or not np.isfinite(growth).all()):
            raise ValueError("A growth policy needs aligned finite public growth observations")
        result["growth"] = torch.as_tensor(growth, device=device)
    return result
