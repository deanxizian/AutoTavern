"""SB3 vector interface to native threaded batches, one learner per lobby."""

import time

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

import taverngym as tg
from .env import TavernGymEnv


class NativeVectorEnv(VecEnv):
    render_mode = None

    def __init__(self, n_envs=256, seed=42, threads=4, config=None):
        if n_envs < 1 or threads < 1:
            raise ValueError("Environment and thread counts must be positive")
        reference = TavernGymEnv(seed=seed)
        self.base_seed, self.threads = seed, threads
        self.config = config
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
        self.batch = tg.BatchEnv(self.num_envs, seed=self.base_seed, controlled=(0,),
                                 threads=self.threads, config=self.config)
        self._returns = np.zeros(self.num_envs, dtype=np.float64)
        self._lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._started = np.full(self.num_envs, time.monotonic())
        self._reset_seeds()
        self._reset_options()
        return self._observe()

    def step_async(self, actions):
        if self._actions is not None:
            raise RuntimeError("A vector step is already pending")
        self._actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs).copy()

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("Call step_async before step_wait")
        actions, self._actions = self._actions, None
        rewards, dones, metadata = self.batch.step(actions)
        if metadata["rejected"].any():
            raise RuntimeError("A native batch rejected a policy action despite masking")
        if ((metadata["placement"] == 0) & dones).any():
            raise RuntimeError("Simulator guard stopped an episode; refusing to train on it")
        self._returns += rewards
        self._lengths += 1
        infos = [{} for _ in range(self.num_envs)]
        now = time.monotonic()
        for i in np.flatnonzero(dones):
            infos[i] = {"placement": int(metadata["placement"][i]),
                        "TimeLimit.truncated": False,
                        "episode": {"r": float(self._returns[i]), "l": int(self._lengths[i]),
                                    "t": now - self._started[i]}}
            self._returns[i], self._lengths[i], self._started[i] = 0, 0, now
        return self._observe(), rewards, dones, infos

    def close(self):
        self.batch = None
        self._obs = None

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
