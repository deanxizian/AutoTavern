"""Gymnasium adapter for one learning seat against seven built-in scripted bots."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import taverngym as tg


class TavernGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, agent=0, config=None):
        self.simulator = tg.BattlegroundsEnv(seed=seed, agent=agent, config=config)
        self.action_space = spaces.Discrete(tg.N_ACTIONS)
        self.observation_space = spaces.Dict({
            "features": spaces.Box(-np.inf, np.inf, (tg.OBS_FLOATS,), dtype=np.float32),
            "cards": spaces.Box(0, tg.N_CARDS - 1, (tg.OBS_CARDS,), dtype=np.int32),
            "action_mask": spaces.MultiBinary(tg.N_ACTIONS),
        })
        self._obs = None
        self._done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._obs, info = self.simulator.reset(seed=seed)
        self._done = False
        return self._obs, info

    def step(self, action):
        self._obs, reward, done, truncated, info = self.simulator.step(int(action))
        self._done = done or truncated
        return self._obs, reward, done, truncated, info

    def action_masks(self):
        if self._obs is None or self._done:
            raise RuntimeError("Reset the environment before requesting an action mask")
        return self._obs["action_mask"].copy()


def make_vector_env(n_envs=8, seed=42):
    from functools import partial
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_one(seed):
        return Monitor(TavernGymEnv(seed=seed))

    if n_envs < 1:
        raise ValueError("n_envs must be positive")
    return DummyVecEnv([partial(make_one, seed + i) for i in range(n_envs)])
