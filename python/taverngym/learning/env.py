"""Gymnasium adapter for one learning seat against seven built-in scripted bots."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import taverngym as tg
from .rewards import ActionRewardShaper, RewardShapingConfig


class TavernGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=0, agent=0, config=None, reward_shaping=None):
        self.simulator = tg.BattlegroundsEnv(seed=seed, agent=agent, config=config)
        self.reward_shaper = (ActionRewardShaper(RewardShapingConfig(**reward_shaping))
                              if reward_shaping is not None else None)
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
        if self.reward_shaper is not None:
            self.reward_shaper.reset()
        return self._obs, info

    def step(self, action):
        before = self._obs
        if self.reward_shaper is not None and self.reward_shaper.action_limit_exceeded(before):
            self._obs, reward, done, truncated, info = self.simulator.forfeit()
            info["termination_reason"] = "action_limit_forfeit"
            reward, components = self.reward_shaper.step(
                action, before, self._obs, reward, done=True,
                terminal_reason="action_limit_forfeit")
            components["action_limit_forfeit_placement"] = info["placement"]
            info["reward_components"] = components
            self._done = True
            return self._obs, reward, done, truncated, info
        self._obs, reward, done, truncated, info = self.simulator.step(int(action))
        self._done = done or truncated
        if self.reward_shaper is not None:
            if info.get("illegal") is not None:
                raise RuntimeError("Reward-shaped training requires a valid native action")
            reward, components = self.reward_shaper.step(
                action, before, self._obs, reward, done=self._done,
            )
            if self.reward_shaper.forfeit_requested:
                self._obs, terminal_reward, done, truncated, info = self.simulator.forfeit()
                info["termination_reason"] = "cycle_forfeit"
                components = self.reward_shaper.finish_forfeit(terminal_reward, info["placement"])
                reward += terminal_reward
                self._done = True
            if components is not None:
                info["reward_components"] = components
        return self._obs, reward, done, truncated, info

    def action_masks(self):
        if self._obs is None or self._done:
            raise RuntimeError("Reset the environment before requesting an action mask")
        return self._obs["action_mask"].copy()


def make_vector_env(n_envs=8, seed=42, reward_shaping=None):
    from functools import partial
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_one(seed):
        return Monitor(TavernGymEnv(seed=seed, reward_shaping=reward_shaping))

    if n_envs < 1:
        raise ValueError("n_envs must be positive")
    return DummyVecEnv([partial(make_one, seed + i) for i in range(n_envs)])
