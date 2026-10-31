"""Role-neutral actor sampling and its PPO vector-environment adapter.

Games advance independently. Ready actors are batched by policy identity, while
only learner actions create transitions. The only barrier is the rollout budget
per game, not every individual learner decision. All observations stay native.
"""

from dataclasses import dataclass
import time
from typing import NamedTuple

import numpy as np
import torch

import taverngym as tg
from .rewards import ActionRewardShaper, RewardShapingConfig
from .vector_env import NativeVectorEnv


class ActorOutput(NamedTuple):
    actions: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray


class PolicyActor:
    """The same inference interface for learner and fixed policy snapshots."""

    def __init__(self, policy):
        self.policy = policy
        self.policy.set_training_mode(False)
        self.batch_sizes = []
        self.inference_seconds = 0.0
        self.last_output = None

    def act(self, obs):
        started = time.monotonic()
        tensors, _ = self.policy.obs_to_tensor(obs)
        with torch.no_grad():
            actions, values, log_probs = self.policy(tensors)
        result = ActorOutput(actions.cpu().numpy(), values.flatten().cpu().numpy(), log_probs.cpu().numpy())
        if not obs["action_mask"][np.arange(len(result.actions)), result.actions].all():
            raise RuntimeError("Actor sampled an action outside the observed native mask")
        self.batch_sizes.append(len(result.actions))
        self.inference_seconds += time.monotonic() - started
        self.last_output = result
        return result

    def predict(self, obs, deterministic=False):
        if deterministic:
            raise ValueError("This sampling prototype uses stochastic actors")
        return self.act(obs).actions, None


@dataclass(slots=True)
class ActorTransition:
    game: int
    episode: int
    episode_step: int
    rollout_step: int
    observation: dict
    action: int
    value: np.float32
    log_prob: np.float32
    reward: np.float32
    done: bool
    placement: int
    next_observation: dict
    info: dict | None = None


@dataclass(slots=True)
class _Pending:
    observation: dict
    action: int
    value: np.float32
    log_prob: np.float32
    reward: float = 0.0


class UnifiedActorSampler:
    def __init__(self, n_games, learner, fixed, *, seed=43, threads=4, config=None,
                 frozen_opponents=1, reward_shaping=None):
        if not 1 <= frozen_opponents <= 7:
            raise ValueError("frozen_opponents must be in 1..7")
        self.n_games, self.roles = n_games, frozen_opponents + 1
        # Separate policy identities even when the two snapshots initially have
        # identical weights. Later PPO updates must never change the fixed bot.
        self.actors = {"learner": learner, "bot_v1": fixed}
        self.batch = tg.BatchEnv(n_games, seed=seed, controlled=tuple(range(self.roles)),
                                 config=config, threads=threads, rotate_seats=True)
        self.observations = self.batch.observe()
        self.episodes = np.zeros(n_games, dtype=np.int64)
        self.episode_steps = np.zeros(n_games, dtype=np.int64)
        self.shapers = ([ActionRewardShaper(RewardShapingConfig(**reward_shaping)) for _ in range(n_games)]
                        if reward_shaping is not None else [])
        self.last_failure = None
        self.config = config or {}
        self.returns = np.zeros(n_games, np.float64)
        self.lengths = np.zeros(n_games, np.int64)
        self.last_turn = np.full(n_games, -1, np.int64)
        self.turn_actions = np.zeros(n_games, np.int64)
        self.turn_count = np.zeros(n_games, np.int64)
        self.cap_ends = np.zeros(n_games, np.int64)
        self.started = np.full(n_games, time.monotonic())

    def _episode_info(self, game, current, next_obs, reward, done, placement):
        turn = int(round(float(current.observation["features"][0]) * 20))
        if turn != self.last_turn[game]:
            self.turn_actions[game] = 0
            self.turn_count[game] += int(turn > 0)
            self.last_turn[game] = turn
        caps = self.config.get("turn_action_caps", [])
        if caps:
            cap = min(caps[min(max(turn - 1, 0), len(caps) - 1)], self.config.get("max_actions_per_turn", 200))
            self.cap_ends[game] += int(current.action == 0 and turn > 0 and self.turn_actions[game] >= cap
                                       and current.observation["action_mask"].sum() == 1)
        self.turn_actions[game] += 1
        info = {}
        if self.shapers:
            reward, components = self.shapers[game].step(
                current.action, current.observation, next_obs, reward, done=done)
            if components is not None:
                components["cap_end_rate"] = float(self.cap_ends[game] / max(self.turn_count[game], 1))
                info["reward_components"] = components
        self.returns[game] += reward
        self.lengths[game] += 1
        if done:
            info.update(placement=placement, **{"TimeLimit.truncated": False},
                        episode={"r": float(self.returns[game]), "l": int(self.lengths[game]),
                                 "t": time.monotonic() - self.started[game]})
            self.returns[game] = self.lengths[game] = 0
            self.last_turn[game] = -1
            self.turn_actions[game] = self.turn_count[game] = self.cap_ends[game] = 0
            self.started[game] = time.monotonic()
        return reward, info

    def learner_observation(self):
        return {key: value[::self.roles].copy() for key, value in self.observations.items() if key != "ready"}

    def collect(self, steps_per_game, on_transition=None, on_batch=None):
        if steps_per_game < 1:
            raise ValueError("steps_per_game must be positive")
        counts = np.zeros(self.n_games, dtype=np.int64)
        pending = [None] * self.n_games
        native_steps = terminal_transitions = submitted = 0
        deferred_opponents = 0
        native_seconds = observation_seconds = reward_seconds = 0.0
        started = time.monotonic()
        while (counts < steps_per_game).any():
            full = self.observations
            active = counts < steps_per_game
            slots = np.arange(self.batch.n_slots)
            ready = full["ready"] & active[slots // self.roles]
            if not ready.any():
                raise RuntimeError("Active rollout has no ready actor")
            actions = np.zeros(self.batch.n_slots, dtype=np.int64)
            for identity, selected in (("learner", ready & (slots % self.roles == 0)),
                                        ("bot_v1", ready & (slots % self.roles != 0))):
                selected = np.flatnonzero(selected)
                if not len(selected):
                    continue
                obs = {key: value[selected] for key, value in full.items() if key != "ready"}
                output = self.actors[identity].act(obs)
                actions[selected] = output.actions
                if identity == "learner":
                    for row, slot in enumerate(selected):
                        game = int(slot // self.roles)
                        if pending[game] is not None:
                            raise RuntimeError("Learner has an unfinished action")
                        pending[game] = _Pending(
                            {key: value[row].copy() for key, value in obs.items()}, int(output.actions[row]),
                            np.float32(output.values[row]), np.float32(output.log_probs[row]))
                        submitted += 1
            tick = time.monotonic()
            rewards, dones, metadata = self.batch.step(actions, active=active)
            try:
                NativeVectorEnv._validate_result(dones, metadata)
            except RuntimeError:
                rejected = np.flatnonzero(metadata["rejected"])
                self.last_failure = {"native_step": native_steps, "rejected_slots": rejected.tolist(),
                    "actions": actions[rejected].tolist(),
                    "legal_in_observed_mask": full["action_mask"][rejected, actions[rejected]].tolist(),
                    "after_states": {str(int(s // self.roles)): self.batch.save(int(s // self.roles)) for s in rejected}}
                raise
            if metadata["deferred"][::self.roles].any():
                raise RuntimeError("First-role learner action unexpectedly deferred")
            deferred_opponents += int(metadata["deferred"].sum())
            self.batch.reset_finished_first()
            native_seconds += time.monotonic() - tick
            tick = time.monotonic()
            after = self.batch.observe()
            observation_seconds += time.monotonic() - tick
            tick = time.monotonic()
            completed = []
            for game in np.flatnonzero(active):
                slot = int(game * self.roles)
                current = pending[game]
                if current is None:
                    raise RuntimeError("Active learner is waiting without a recorded action")
                current.reward += float(rewards[slot])
                done = bool(dones[slot])
                if not done and not after["ready"][slot]:
                    continue
                next_obs = {key: value[slot] for key, value in after.items() if key != "ready"}
                placement = int(metadata["placement"][slot]) if done else 0
                reward, info = self._episode_info(game, current, next_obs, current.reward, done, placement)
                transition = ActorTransition(
                    int(game), int(self.episodes[game]), int(self.episode_steps[game]), int(counts[game]),
                    current.observation, current.action, current.value, current.log_prob,
                    np.float32(reward), done, placement, next_obs, info)
                if on_transition is not None:
                    on_transition(transition)
                if on_batch is not None:
                    completed.append(transition)
                counts[game] += 1
                self.episode_steps[game] += 1
                if done:
                    terminal_transitions += 1
                    self.episodes[game] += 1
                    self.episode_steps[game] = 0
                pending[game] = None
            if completed:
                on_batch(completed)
            reward_seconds += time.monotonic() - tick
            self.observations = after
            native_steps += 1
            if native_steps > max(10_000, steps_per_game * 200):
                raise RuntimeError("Actor collection exceeded its bounded native-step budget")
        assert submitted == int(counts.sum()) == self.n_games * steps_per_game
        assert all(item is None for item in pending)
        assert self.observations["ready"][::self.roles].all()
        seconds = time.monotonic() - started
        return {"learner_samples": submitted, "sampling_seconds": seconds,
                "learner_samples_per_second": submitted / seconds, "native_steps": native_steps,
                "terminal_transitions": terminal_transitions, "native_seconds": native_seconds,
                "deferred_opponent_actions": deferred_opponents,
                "observation_seconds": observation_seconds, "reward_and_sink_seconds": reward_seconds}


class ActorOpponentVectorEnv(NativeVectorEnv):
    """PPO owns inference scheduling; standard single vector steps are unavailable."""

    def __init__(self, *args, opponent_policy, frozen_opponents=2, **kwargs):
        self.opponent_policy, self.frozen_opponents = opponent_policy, frozen_opponents
        self.opponent_decisions = 0
        self.sampler = None
        super().__init__(*args, **kwargs)

    def _create_batch(self):
        costs = (vars(self.reward_shaping) if self.reward_shaping is not None else None)
        self.sampler = UnifiedActorSampler(self.num_envs, None, PolicyActor(self.opponent_policy),
            seed=self.base_seed, threads=self.threads, config=self.config,
            frozen_opponents=self.frozen_opponents, reward_shaping=costs)
        return self.sampler.batch

    def _observe(self):
        self._obs = self.sampler.learner_observation()
        return self._obs

    def collect_actors(self, policy, steps, sink):
        policy.set_training_mode(False)
        self.sampler.actors = {"learner": PolicyActor(policy), "bot_v1": PolicyActor(self.opponent_policy)}
        result = self.sampler.collect(steps, on_batch=sink)
        self.opponent_decisions += sum(self.sampler.actors["bot_v1"].batch_sizes) - result["deferred_opponent_actions"]
        self._observe()
        for identity, actor in self.sampler.actors.items():
            result[f"{identity}_forward_calls"] = len(actor.batch_sizes)
            result[f"{identity}_batch_mean"] = float(np.mean(actor.batch_sizes))
            result[f"{identity}_batch_median"] = float(np.median(actor.batch_sizes))
            result[f"{identity}_inference_seconds"] = actor.inference_seconds
        return result

    def step_wait(self):
        raise RuntimeError("Actor sampling requires EntityPPO.collect_rollouts; evaluate with a separate vector environment")

    def close(self):
        self.sampler = None
        super().close()
