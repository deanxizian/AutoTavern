"""Lossless disk-backed rollouts; sampling never flattens/copies the complete buffer."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import taverngym as tg
from .model import ObservationLayout

from .ppo import PlacementBuffer, PlacementSamples


class DiskPlacementBuffer(PlacementBuffer):
    def __init__(self, *args, storage_root="/tmp/autotavern-ppo-rollouts", **kwargs):
        root = Path(storage_root)
        root.mkdir(parents=True, exist_ok=True)
        self._temporary = TemporaryDirectory(prefix="rollout-", dir=root)
        self._mapped = {}
        super().__init__(*args, **kwargs)

    def reset(self):
        # No native reset: it would allocate >50 GiB in RAM before replacing arrays.
        shape = (self.buffer_size, self.n_envs)
        self.pos, self.full, self.generator_ready = 0, False, False
        self.observations = self._mapped
        for name in ("rewards", "episode_starts", "values", "log_probs", "advantages", "returns"):
            setattr(self, name, np.zeros(shape, dtype=np.float32))
        self.actions = np.zeros((*shape, self.action_dim), dtype=np.int64)
        self.placement_targets = np.full(shape, -1, dtype=np.int64)
        self.segment_starts = np.zeros(self.n_envs, dtype=np.int64)
        self.mask_dims = int(self.action_space.n)
        self.action_masks = self._mapped.get("action_mask")

    def _allocate(self):
        if self._mapped:
            return
        for key, shape in self.obs_shape.items():
            self._mapped[key] = np.memmap(
                Path(self._temporary.name) / f"{key}.bin", mode="w+",
                dtype=self.observation_space[key].dtype,
                shape=(self.buffer_size, self.n_envs, *shape),
            )
        self.action_masks = self._mapped["action_mask"]

    def add(self, *args, **kwargs):
        self._allocate()
        super().add(*args, **kwargs)

    def get(self, batch_size=None):
        if not self.full:
            raise RuntimeError("Cannot optimize a partial rollout")
        count = self.buffer_size * self.n_envs
        indices = np.random.permutation(count)
        size = batch_size or count
        for begin in range(0, count, size):
            yield self._get_samples(indices[begin:begin + size])

    def _get_samples(self, batch_inds, env=None):
        # Match SB3's environment-major flatten ordering without creating a full copy.
        environments, steps = np.divmod(batch_inds, self.buffer_size)
        at = (steps, environments)
        obs = {key: self.to_torch(value[at]) for key, value in self.observations.items()}
        return PlacementSamples(
            obs, self.to_torch(self.actions[at]), self.to_torch(self.values[at]),
            self.to_torch(self.log_probs[at]), self.to_torch(self.advantages[at]),
            self.to_torch(self.returns[at]), obs["action_mask"],
            self.to_torch(self.placement_targets[at]).long(),
        )

    def close(self):
        self.action_masks = None
        for mapped in self._mapped.values():
            mapped.flush()
            mapped._mmap.close()
        self._mapped.clear()
        self._temporary.cleanup()


class ObservationCodec:
    """Exact float32 values for numerics, bit packing for schema-defined binary flags."""

    def __init__(self):
        layout = ObservationLayout()
        n = layout.native
        if (n["MINION_F"], n["ITEM_F"], n["GLOBAL_F"]) != (31, 40, 53):
            raise ValueError("Observation codec needs updating for this native schema")
        binary = [1, 2, 4, 15, 16, 19, 22, 27, 28, 35, 37, *range(42, 53)]
        binary.extend(n["GLOBAL_F"] + 2 * i for i in range(n["N_QUESTS"]))
        pending = n["GLOBAL_F"] + n["N_QUESTS"] * n["QUEST_F"]
        binary.extend(range(pending, pending + n["PENDING_KINDS"]))
        minion_bits = [0, 5, *range(8, n["MINION_F"])]
        item_bits = [0, 1, 4, 5, 6, 7, 8, *(9 + i for i in minion_bits)]
        for name in ("board", "hand", "shop", "choices", "opponents", "opponent_boards"):
            span, count, width = layout.floats[name]
            bits = ([0, 1, 7, 8] if name == "opponents" else
                    minion_bits if name in ("board", "opponent_boards") else item_bits)
            binary.extend(span.start + i * width + column for i in range(count) for column in bits)
        self.binary = np.array(sorted(binary), dtype=np.int64)
        self.numeric = np.setdiff1d(np.arange(tg.OBS_FLOATS), self.binary)
        self.flag_bytes = (len(self.binary) + 7) // 8
        self.mask_bytes = (tg.N_ACTIONS + 7) // 8
        if len(np.unique(self.binary)) != len(self.binary):
            raise ValueError("Duplicate binary observation columns")
        if tg.N_CARDS > np.iinfo(np.uint16).max + 1:
            raise ValueError("Card vocabulary no longer fits in uint16")

    def encode(self, obs):
        flags = obs["features"][..., self.binary]
        if not np.all((flags == 0) | (flags == 1)):
            raise ValueError("A schema-defined binary observation feature is not 0 or 1")
        cards = obs["cards"]
        if np.any(cards < 0) or np.any(cards >= tg.N_CARDS):
            raise ValueError("Invalid card ID in rollout observation")
        return (obs["features"][..., self.numeric].astype(np.float32),
                np.packbits(flags.astype(np.uint8), axis=-1, bitorder="little"),
                cards.astype(np.uint16),
                np.packbits(obs["action_mask"], axis=-1, bitorder="little"))

    def decode(self, numerics, flags, cards, masks):
        features = np.empty((*numerics.shape[:-1], tg.OBS_FLOATS), dtype=np.float32)
        features[..., self.numeric] = numerics
        features[..., self.binary] = np.unpackbits(flags, axis=-1, count=len(self.binary), bitorder="little")
        return {"features": features, "cards": cards.astype(np.int32),
                "action_mask": np.unpackbits(masks, axis=-1, count=tg.N_ACTIONS, bitorder="little").astype(bool)}

    def bytes_per_observation(self):
        return len(self.numeric) * 4 + self.flag_bytes + tg.OBS_CARDS * 2 + self.mask_bytes


class CompactPlacementBuffer(PlacementBuffer):
    """Lossless RAM rollouts: ~13 GiB for 4,194,304 observations, no full flatten copy."""

    def __init__(self, *args, **kwargs):
        self.codec = ObservationCodec()
        self._storage = None
        super().__init__(*args, **kwargs)

    def reset(self):
        shape = (self.buffer_size, self.n_envs)
        self.pos, self.full, self.generator_ready = 0, False, False
        for name in ("rewards", "episode_starts", "values", "log_probs", "advantages", "returns"):
            setattr(self, name, np.zeros(shape, dtype=np.float32))
        self.actions = np.zeros((*shape, self.action_dim), dtype=np.int64)
        self.placement_targets = np.full(shape, -1, dtype=np.int64)
        self.segment_starts = np.zeros(self.n_envs, dtype=np.int64)
        self.actor_counts = np.zeros(self.n_envs, dtype=np.int64)
        self.mask_dims = int(self.action_space.n)
        if self._storage is None:
            self._storage = (
                np.empty((*shape, len(self.codec.numeric)), dtype=np.float32),
                np.empty((*shape, self.codec.flag_bytes), dtype=np.uint8),
                np.empty((*shape, tg.OBS_CARDS), dtype=np.uint16),
                np.empty((*shape, self.codec.mask_bytes), dtype=np.uint8),
            )

    def add(self, obs, action, reward, episode_start, value, log_prob, action_masks=None):
        if action_masks is not None and not np.array_equal(obs["action_mask"], action_masks):
            raise ValueError("Stored action mask differs from the policy observation")
        for storage, encoded in zip(self._storage, self.codec.encode(obs)):
            storage[self.pos] = encoded
        self.actions[self.pos] = action.reshape(self.n_envs, self.action_dim)
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.detach().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.detach().cpu().numpy().flatten()
        self.pos += 1
        self.full = self.pos == self.buffer_size

    def add_actor_batch(self, transitions):
        """Scatter independently completed learner decisions into [step, game]."""
        games = np.array([t.game for t in transitions], dtype=np.int64)
        steps = np.array([t.rollout_step for t in transitions], dtype=np.int64)
        if len(np.unique(games)) != len(games) or not np.array_equal(self.actor_counts[games], steps):
            raise ValueError("Actor samples must be consecutive and unique per game")
        if np.any(steps >= self.buffer_size):
            raise ValueError("Actor sample exceeds rollout budget")
        obs = {key: np.stack([t.observation[key] for t in transitions]) for key in transitions[0].observation}
        at = (steps, games)
        for storage, encoded in zip(self._storage, self.codec.encode(obs)):
            storage[at] = encoded
        self.actions[steps, games, 0] = [t.action for t in transitions]
        for field, values in (("rewards", [t.reward for t in transitions]),
                              ("values", [t.value for t in transitions]),
                              ("log_probs", [t.log_prob for t in transitions]),
                              ("episode_starts", [t.episode_step == 0 for t in transitions])):
            getattr(self, field)[at] = values
        for t in transitions:
            if t.done:
                if not 1 <= t.placement <= 8:
                    raise ValueError("Terminal actor transition requires a real placement")
                self.placement_targets[self.segment_starts[t.game]:t.rollout_step + 1, t.game] = t.placement - 1
                self.segment_starts[t.game] = t.rollout_step + 1
        self.actor_counts[games] += 1
        self.pos = int(self.actor_counts.min())
        self.full = bool((self.actor_counts == self.buffer_size).all())

    def get(self, batch_size=None):
        if not self.full:
            raise RuntimeError("Cannot optimize a partial rollout")
        total = self.buffer_size * self.n_envs
        indices = np.random.permutation(total)
        size = batch_size or total
        for begin in range(0, total, size):
            yield self._get_samples(indices[begin:begin + size])

    def _get_samples(self, batch_inds, env=None):
        environments, steps = np.divmod(batch_inds, self.buffer_size)
        at = (steps, environments)
        obs = {key: self.to_torch(value) for key, value in self.codec.decode(
            *(storage[at] for storage in self._storage)).items()}
        return PlacementSamples(
            obs, self.to_torch(self.actions[at]), self.to_torch(self.values[at]),
            self.to_torch(self.log_probs[at]), self.to_torch(self.advantages[at]),
            self.to_torch(self.returns[at]), obs["action_mask"],
            self.to_torch(self.placement_targets[at]).long(),
        )

    def close(self):
        self._storage = None
