"""SB3 vector interface to native threaded batches, one learner per lobby."""

import time

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

import taverngym as tg
from .env import TavernGymEnv
from .rewards import ActionRewardShaper, RewardShapingConfig


class NativeVectorEnv(VecEnv):
    render_mode = None

    def __init__(self, n_envs=256, seed=42, threads=4, config=None, reward_shaping=None, rotate_seats=False):
        if n_envs < 1 or threads < 1:
            raise ValueError("Environment and thread counts must be positive")
        reference = TavernGymEnv(seed=seed)
        self.base_seed, self.threads = seed, threads
        self.config = config
        self.rotate_seats = rotate_seats
        self.reward_shaping = RewardShapingConfig(**reward_shaping) if reward_shaping is not None else None
        self._reward_shapers = []
        self.batch = None
        self._obs = None
        self._actions = None
        super().__init__(n_envs, reference.observation_space, reference.action_space)

    def _observe(self):
        obs = self.batch.observe()
        if not obs.pop("ready").all():
            raise RuntimeError("A single-controlled-seat batch must always have a ready learner")
        self._obs = obs
        return obs

    def reset(self):
        if self._seeds[0] is not None:
            self.base_seed = self._seeds[0]
        self.batch = self._create_batch()
        self._returns = np.zeros(self.num_envs, dtype=np.float64)
        self._lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._started = np.full(self.num_envs, time.monotonic())
        self._last_turn = np.full(self.num_envs, -1, dtype=np.int64)
        self._turn_actions = np.zeros(self.num_envs, dtype=np.int64)
        self._turn_count = np.zeros(self.num_envs, dtype=np.int64)
        self._cap_ends = np.zeros(self.num_envs, dtype=np.int64)
        self._reward_shapers = ([ActionRewardShaper(self.reward_shaping) for _ in range(self.num_envs)]
                                if self.reward_shaping is not None else [])
        self._reset_seeds()
        self._reset_options()
        return self._observe()

    def _create_batch(self):
        return tg.BatchEnv(self.num_envs, seed=self.base_seed, controlled=(0,),
                           threads=self.threads, config=self.config, rotate_seats=self.rotate_seats)

    def step_async(self, actions):
        if self._actions is not None:
            raise RuntimeError("A vector step is already pending")
        self._actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs).copy()

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("Call step_async before step_wait")
        actions, self._actions = self._actions, None
        before = self._obs
        rewards, dones, metadata = self.batch.step(actions)
        self._validate_result(dones, metadata)
        return self._finish_transition(before, self._observe(), actions, rewards, dones, metadata)

    @staticmethod
    def _validate_result(dones, metadata):
        if metadata["rejected"].any():
            raise RuntimeError("A native batch rejected a policy action despite masking")
        if ((metadata["placement"] == 0) & dones).any():
            raise RuntimeError("Simulator guard stopped an episode; refusing to train on it")

    def _finish_transition(self, before, after, actions, rewards, dones, metadata):
        turn = np.rint(before["features"][:, 0] * 20).astype(np.int64)
        changed = turn != self._last_turn
        self._turn_actions[changed] = 0
        self._turn_count += changed & (turn > 0)
        self._last_turn = turn
        caps = (self.config or {}).get("turn_action_caps", [])
        if caps:
            cap = np.minimum(np.asarray(caps)[np.minimum(np.maximum(turn - 1, 0), len(caps) - 1)],
                             (self.config or {}).get("max_actions_per_turn", 200))
            self._cap_ends += ((actions == 0) & (turn > 0) & (self._turn_actions >= cap)
                               & (before["action_mask"].sum(axis=1) == 1))
        self._turn_actions += 1
        infos = [{} for _ in range(self.num_envs)]
        for i, shaper in enumerate(self._reward_shapers):
            rewards[i], components = shaper.step(
                actions[i], {key: value[i] for key, value in before.items()},
                {key: value[i] for key, value in after.items()}, rewards[i], done=bool(dones[i]),
            )
            if components is not None:
                infos[i]["reward_components"] = components
        self._returns += rewards
        self._lengths += 1
        now = time.monotonic()
        for i in np.flatnonzero(dones):
            if "reward_components" in infos[i]:
                infos[i]["reward_components"]["cap_end_rate"] = float(
                    self._cap_ends[i] / max(self._turn_count[i], 1))
            infos[i].update({"placement": int(metadata["placement"][i]),
                             "TimeLimit.truncated": False,
                             "episode": {"r": float(self._returns[i]), "l": int(self._lengths[i]),
                                         "t": now - self._started[i]}})
            self._returns[i], self._lengths[i], self._started[i] = 0, 0, now
            self._last_turn[i] = -1
            self._turn_actions[i] = self._turn_count[i] = self._cap_ends[i] = 0
        return after, rewards, dones, infos

    def close(self):
        self.batch = None
        self._obs = None
        self._reward_shapers = []

    def get_attr(self, attr_name, indices=None):
        selected = self._get_indices(indices)
        if attr_name == "render_mode":
            return [None for _ in selected]
        if attr_name == "action_masks":
            return [self.action_masks for _ in selected]
        if attr_name in ("action_space", "observation_space", "metadata"):
            return [getattr(self, attr_name) for _ in selected]
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        raise AttributeError(f"Native vector attributes are immutable: {attr_name}")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if method_name != "action_masks":
            raise AttributeError(method_name)
        return [self._obs["action_mask"][i].copy() for i in self._get_indices(indices)]

    def action_masks(self):
        return self._obs["action_mask"].copy()

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class MixedOpponentVectorEnv(NativeVectorEnv):
    """One learner vs fixed-policy roles plus native bots, with per-game seat rotation.

    Every returned transition contains exactly one actual learner action. While
    other games catch up, ready learners and their entire games are held unchanged.
    If a learner dies early, its placement is already final: reset that game
    immediately. Opponent actions never enter the learner's buffer.
    """

    def __init__(self, *args, opponent_policy, frozen_opponents=4, **kwargs):
        if not 1 <= frozen_opponents <= 7:
            raise ValueError("frozen_opponents must be in 1..7")
        self.opponent_policy = opponent_policy
        self.frozen_opponents = frozen_opponents
        self.roles = frozen_opponents + 1
        self.opponent_decisions = 0
        self.waiting_rounds = 0
        self._all_obs = None
        super().__init__(*args, **kwargs)

    def _create_batch(self):
        return tg.BatchEnv(self.num_envs, seed=self.base_seed, controlled=tuple(range(self.roles)),
                           threads=self.threads, config=self.config, rotate_seats=True)

    def _set_observation(self, full):
        self._all_obs = full
        if not full["ready"][::self.roles].all():
            raise RuntimeError("Mixed driver returned a learner that is not ready")
        self._obs = {key: value[::self.roles].copy() for key, value in full.items() if key != "ready"}
        return self._obs

    def _observe(self):
        return self._set_observation(self.batch.observe())

    def _advance(self, full, active, learner_actions=None):
        indices = np.arange(self.batch.n_slots)
        opponent_slots = np.flatnonzero(full["ready"] & active[indices // self.roles]
                                        & (indices % self.roles != 0))
        actions = np.zeros(self.batch.n_slots, dtype=np.int64)
        if learner_actions is not None:
            actions[::self.roles] = learner_actions
        if len(opponent_slots):
            obs = {key: value[opponent_slots] for key, value in full.items() if key != "ready"}
            predicted, _ = self.opponent_policy.predict(obs, deterministic=False)
            actions[opponent_slots] = predicted
            self.opponent_decisions += len(opponent_slots)
        rewards, dones, metadata = self.batch.step(actions, active=active)
        self._validate_result(dones, metadata)
        self.opponent_decisions -= int(metadata["deferred"][opponent_slots].sum())
        self.batch.reset_finished_first()
        return rewards[::self.roles], dones[::self.roles], metadata["placement"][::self.roles]

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("Call step_async before step_wait")
        actions, self._actions = self._actions, None
        before = self._obs
        learner_seats = self.batch.slot_seats[::self.roles].copy()
        rewards, dones, placement = self._advance(self._all_obs, np.ones(self.num_envs, dtype=bool), actions)
        # Slices must be independent of later native step outputs.
        rewards, dones, placement = rewards.copy(), dones.copy(), placement.copy()
        for _ in range(10_000):
            full = self.batch.observe()
            active = ~full["ready"][::self.roles]
            if not active.any():
                break
            self.waiting_rounds += 1
            extra_reward, extra_done, extra_placement = self._advance(full, active)
            rewards += extra_reward
            placement[extra_done] = extra_placement[extra_done]
            dones |= extra_done
        else:
            raise RuntimeError("Mixed opponents failed to reach the next learner decision")
        after = self._set_observation(full)
        metadata = {"placement": placement, "rejected": np.zeros(self.num_envs, dtype=bool)}
        result = self._finish_transition(before, after, actions, rewards, dones, metadata)
        for i in np.flatnonzero(dones):
            result[3][i]["learner_seat"] = int(learner_seats[i])
        return result

    def close(self):
        super().close()
        self._all_obs = None
