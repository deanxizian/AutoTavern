"""SB3 vector interface to native threaded batches, one learner per lobby."""

import time

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

import taverngym as tg
from .env import TavernGymEnv
from .rewards import ActionRewardShaper, RewardShapingConfig
from .opponents import opponent_groups


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
        self._clear_restored_histories()
        obs = self.batch.observe()
        if not obs.pop("ready").all():
            raise RuntimeError("A single-controlled-seat batch must always have a ready learner")
        self._obs = obs
        return obs

    def _clear_restored_histories(self):
        versions = self.batch._restore_versions
        previous = getattr(self, "_restore_versions", versions)
        for game in np.flatnonzero(previous != versions):
            if self._reward_shapers:
                self._reward_shapers[game].reset()
        self._restore_versions = versions.copy()

    def reset(self):
        if self._seeds[0] is not None:
            self.base_seed = self._seeds[0]
        self.batch = self._create_batch()
        self._restore_versions = self.batch._restore_versions.copy()
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
        rewards, dones, metadata = self.batch.step(actions, forfeit=self._budget_forfeits(before))
        self._validate_result(dones, metadata)
        return self._finish_transition(before, self._observe(), actions, rewards, dones, metadata)

    def _budget_forfeits(self, before):
        selected = np.zeros(self.batch.n_slots, dtype=bool)
        roles = getattr(self, "roles", 1)
        for i, shaper in enumerate(self._reward_shapers):
            selected[i * roles] = shaper.action_limit_exceeded(
                {key: value[i] for key, value in before.items()})
        return selected

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
                               & ~metadata.get("forfeited", np.zeros(self.num_envs, dtype=bool))
                               & (before["action_mask"].sum(axis=1) == 1))
        self._turn_actions += 1
        infos = [{} for _ in range(self.num_envs)]
        for i, shaper in enumerate(self._reward_shapers):
            forfeited = metadata.get("forfeited")
            reason = "action_limit_forfeit" if forfeited is not None and forfeited[i] else None
            rewards[i], components = shaper.step(
                actions[i], {key: value[i] for key, value in before.items()},
                {key: value[i] for key, value in after.items()}, rewards[i], done=bool(dones[i]),
                terminal_reason=reason,
            )
            if components is not None:
                if reason:
                    components["action_limit_forfeit_placement"] = int(metadata["placement"][i])
                    infos[i]["termination_reason"] = reason
                infos[i]["reward_components"] = components
        forfeiting = [i for i, shaper in enumerate(self._reward_shapers)
                      if shaper.forfeit_requested]
        if forfeiting:
            roles = getattr(self, "roles", 1)
            selected = np.zeros(self.batch.n_slots, dtype=bool)
            selected[np.asarray(forfeiting) * roles] = True
            terminal_rewards, terminal_dones, terminal_info = self.batch.forfeit(selected)
            self._validate_result(terminal_dones, terminal_info)
            for i in forfeiting:
                slot = i * roles
                if not terminal_dones[slot]:
                    raise RuntimeError("Native cycle surrender failed to end the learner episode")
                place = int(terminal_info["placement"][slot])
                infos[i]["reward_components"] = self._reward_shapers[i].finish_forfeit(
                    terminal_rewards[slot], place)
                infos[i]["termination_reason"] = "cycle_forfeit"
                metadata["placement"][i] = place
                rewards[i] += terminal_rewards[slot]
                dones[i] = True
            self.batch.reset_finished_first()
            after = self._observe()
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
        self.opponent_groups = opponent_groups(opponent_policy, frozen_opponents)
        self._role_aware = any(group.reward_shaping is not None for group in self.opponent_groups)
        self._seat_sampler = None
        self.frozen_opponents = frozen_opponents
        self.roles = frozen_opponents + 1
        self.role_names = np.array(["learner", *(group.name for group in self.opponent_groups
                                                for _ in range(group.count))])
        self.opponent_decisions = 0
        self.waiting_rounds = 0
        self._all_obs = None
        super().__init__(*args, **kwargs)

    def _create_batch(self):
        if self._role_aware:
            from .actor_sampling import UnifiedActorSampler
            from .opponents import FrozenOpponentPool, OpponentGroup
            from .role_sampling import PredictActor

            policies = FrozenOpponentPool(tuple(OpponentGroup(group.name,
                PredictActor(group.policy), group.count, group.reward_shaping)
                for group in self.opponent_groups))
            self._seat_sampler = UnifiedActorSampler(self.num_envs, None, policies,
                seed=self.base_seed, threads=self.threads, config=self.config,
                frozen_opponents=self.frozen_opponents,
                reward_shaping=vars(self.reward_shaping) if self.reward_shaping else None)
            return self._seat_sampler.batch
        return tg.BatchEnv(self.num_envs, seed=self.base_seed, controlled=tuple(range(self.roles)),
                           threads=self.threads, config=self.config, rotate_seats=True)

    def _set_observation(self, full):
        self._all_obs = full
        if not full["ready"][::self.roles].all():
            raise RuntimeError("Mixed driver returned a learner that is not ready")
        self._obs = {key: value[::self.roles].copy() for key, value in full.items() if key != "ready"}
        return self._obs

    def _observe(self):
        self._clear_restored_histories()
        if self._seat_sampler is not None:
            self._seat_sampler.sync_restored_history()
        return self._set_observation(self.batch.observe())

    def _advance(self, full, active, learner_actions=None):
        indices = np.arange(self.batch.n_slots)
        cap_flags = self._budget_forfeits(self._obs) if learner_actions is not None else None
        forfeiting_games = (cap_flags[::self.roles] if cap_flags is not None
                            else np.zeros(self.num_envs, dtype=bool))
        opponent_slots = np.flatnonzero(full["ready"] & active[indices // self.roles]
                                        & ~forfeiting_games[indices // self.roles]
                                        & (indices % self.roles != 0))
        actions = np.zeros(self.batch.n_slots, dtype=np.int64)
        if learner_actions is not None:
            actions[::self.roles] = learner_actions
        for group in self.opponent_groups:
            selected = opponent_slots[self.role_names[opponent_slots % self.roles] == group.name]
            if not len(selected):
                continue
            obs = {key: value[selected] for key, value in full.items() if key != "ready"}
            predicted, _ = group.policy.predict(obs, deterministic=False)
            actions[selected] = predicted
            self.opponent_decisions += len(selected)
        rewards, dones, metadata = self.batch.step(actions, active=active, forfeit=cap_flags)
        self._validate_result(dones, metadata)
        self.opponent_decisions -= int(metadata["deferred"][opponent_slots].sum())
        self.batch.reset_finished_first()
        forfeited = metadata.get("forfeited", np.zeros(self.batch.n_slots, dtype=bool))
        return (rewards[::self.roles], dones[::self.roles], metadata["placement"][::self.roles],
                forfeited[::self.roles])

    def step_wait(self):
        if self._actions is None:
            raise RuntimeError("Call step_async before step_wait")
        actions, self._actions = self._actions, None
        if self._role_aware:
            from .role_sampling import ExternalActionActor

            self._seat_sampler.actors["learner"] = ExternalActionActor(actions)
            transitions = []
            stats = self._seat_sampler.collect(1, transitions.append)
            transitions.sort(key=lambda transition: transition.game)
            self._all_obs = self._seat_sampler.observations
            self._obs = self._seat_sampler.learner_observation()
            self._reward_shapers = self._seat_sampler.shapers
            self._seat_sampler.deferred_requests = (
                getattr(self._seat_sampler, "deferred_requests", 0)
                + stats["deferred_opponent_actions"])
            self.opponent_decisions = sum(actor.requests for identity, actor in
                self._seat_sampler.actors.items() if identity != "learner") - (
                    self._seat_sampler.deferred_requests)
            return (self._obs, np.array([t.reward for t in transitions], np.float32),
                    np.array([t.done for t in transitions]), [t.info for t in transitions])
        before = self._obs
        learner_seats = self.batch.slot_seats[::self.roles].copy()
        rewards, dones, placement, forfeited = self._advance(
            self._all_obs, np.ones(self.num_envs, dtype=bool), actions)
        # Slices must be independent of later native step outputs.
        rewards, dones, placement = rewards.copy(), dones.copy(), placement.copy()
        for _ in range(10_000):
            full = self.batch.observe()
            active = ~full["ready"][::self.roles]
            if not active.any():
                break
            self.waiting_rounds += 1
            extra_reward, extra_done, extra_placement, _ = self._advance(full, active)
            rewards += extra_reward
            placement[extra_done] = extra_placement[extra_done]
            dones |= extra_done
        else:
            raise RuntimeError("Mixed opponents failed to reach the next learner decision")
        after = self._set_observation(full)
        metadata = {"placement": placement, "rejected": np.zeros(self.num_envs, dtype=bool),
                    "forfeited": forfeited}
        result = self._finish_transition(before, after, actions, rewards, dones, metadata)
        for i in np.flatnonzero(dones):
            result[3][i]["learner_seat"] = int(learner_seats[i])
        return result

    def close(self):
        super().close()
        self._all_obs = None
        self._seat_sampler = None
