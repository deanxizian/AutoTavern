"""Masked PPO with placement supervision from completed rollout segments.

The PPO objective follows sb3-contrib's MaskablePPO (MIT). Placement labels are
training targets only, never observation features. Unfinished segments are ignored.
The first stage has no previous-stage KL anchor.
"""

from typing import NamedTuple
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


DEFAULT_MICRO_BATCH_SIZE = 1024


class PlacementSamples(NamedTuple):
    observations: dict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    action_masks: torch.Tensor
    placement_targets: torch.Tensor


class PlacementBuffer(MaskableDictRolloutBuffer):
    def reset(self):
        super().reset()
        self.placement_targets = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self.segment_starts = np.zeros(self.n_envs, dtype=np.int64)

    def record_endings(self, dones, infos):
        # Called before add(), so pos is the transition whose action just ended.
        for env_index, done in enumerate(dones):
            if done:
                place = infos[env_index].get("placement", 0)
                if 1 <= place <= 8 and not infos[env_index].get("TimeLimit.truncated", False):
                    self.placement_targets[self.segment_starts[env_index]:self.pos + 1, env_index] = place - 1
                self.segment_starts[env_index] = self.pos + 1

    def get(self, batch_size=None):
        if not self.generator_ready:
            self.placement_targets = self.swap_and_flatten(self.placement_targets).flatten()
        yield from super().get(batch_size)

    def _get_samples(self, batch_inds, env=None):
        sample = super()._get_samples(batch_inds, env)
        return PlacementSamples(*sample, self.to_torch(self.placement_targets[batch_inds]).long())


class _PlacementRecorder(BaseCallback):
    def __init__(self, delegate):
        super().__init__()
        self.delegate = delegate

    def _on_rollout_start(self):
        self.delegate.on_rollout_start()

    def _on_step(self):
        self.locals["rollout_buffer"].record_endings(self.locals["dones"], self.locals["infos"])
        self.model._rollout_places.extend(
            info["placement"] for done, info in zip(self.locals["dones"], self.locals["infos"])
            if done and 1 <= info.get("placement", 0) <= 8
        )
        self.model._rollout_reward_components.extend(
            info["reward_components"] for done, info in zip(self.locals["dones"], self.locals["infos"])
            if done and "reward_components" in info
        )
        self.delegate.update_locals(self.locals)
        return self.delegate.on_step()

    def _on_rollout_end(self):
        self.delegate.update_locals(self.locals)
        self.delegate.on_rollout_end()


class EntityPPO(MaskablePPO):
    def __init__(self, *args, aux_coef=0.1, micro_batch_size=DEFAULT_MICRO_BATCH_SIZE, **kwargs):
        if aux_coef < 0:
            raise ValueError("aux_coef must be nonnegative")
        self.aux_coef = aux_coef
        self.micro_batch_size = micro_batch_size
        kwargs.setdefault("rollout_buffer_class", PlacementBuffer)
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            "_after_update", "_after_optimizer_step", "_rollout_places", "_rollout_reward_components",
        ]

    def optimizer_total_step_count(self):
        """Read Adam's persisted counter on the shared, always-used identity table."""
        state = self.policy.optimizer.state.get(self.policy.network.identity.weight, {})
        value = state.get("step", 0)
        return int(value.item()) if torch.is_tensor(value) else int(value)

    def optimizer_step_count(self):
        """Actual updates in this stage; the underlying Adam clock may be older."""
        return self.optimizer_total_step_count() - getattr(self, "_optimizer_step_offset", 0)

    def set_forward_precision(self, precision):
        self.policy.set_forward_precision(precision)
        # SB3 reconstructs the policy from this dict when loading a full model.
        self.policy_kwargs["forward_precision"] = self.policy.forward_precision

    def start_new_stage(self, *, preserve_adam=False):
        """Reset local counters; optionally retain Adam moments and its global clock."""
        if preserve_adam and not self.policy.optimizer.state:
            raise ValueError("Preserving Adam requires a trained checkpoint with optimizer state")
        total_steps = self.optimizer_total_step_count()
        self.policy.optimizer.zero_grad(set_to_none=True)
        if not preserve_adam:
            self.policy.optimizer.state.clear()
        self._optimizer_step_offset = total_steps if preserve_adam else 0
        self.num_timesteps = self._n_updates = self._episode_num = 0
        self._total_timesteps = self._num_timesteps_at_start = 0
        self._last_obs = self._last_original_obs = None
        self.ep_info_buffer = self.ep_success_buffer = None

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True):
        started = time.monotonic()
        self._rollout_places = []
        self._rollout_reward_components = []
        opponent_before = getattr(env, "opponent_decisions", 0)
        if hasattr(env, "collect_actors"):
            result = self._collect_actor_rollouts(env, callback, rollout_buffer, n_rollout_steps)
        else:
            recorder = _PlacementRecorder(callback)
            recorder.init_callback(self)
            result = super().collect_rollouts(env, recorder, rollout_buffer, n_rollout_steps, use_masking)
        if self._rollout_reward_components:
            for name in self._rollout_reward_components[0]:
                self.logger.record(f"rollout/{name}", float(np.mean([
                    row[name] for row in self._rollout_reward_components
                ])))
        self.last_rollout_seconds = time.monotonic() - started
        if hasattr(env, "opponent_decisions"):
            self.logger.record("rollout/opponent_decisions", env.opponent_decisions - opponent_before)
        return result

    def _collect_actor_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        if not hasattr(rollout_buffer, "add_actor_batch") or rollout_buffer.buffer_size != n_rollout_steps:
            raise ValueError("Actor rollout requires a matching CompactPlacementBuffer")
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        last_dones = np.zeros(env.num_envs, dtype=bool)
        callback.on_rollout_start()

        def consume(transitions):
            rollout_buffer.add_actor_batch(transitions)
            self.num_timesteps += len(transitions)
            dones = np.array([t.done for t in transitions])
            infos = [t.info or {} for t in transitions]
            for t in transitions:
                last_dones[t.game] = t.done
                if t.done:
                    self._rollout_places.append(t.placement)
                    if "reward_components" in (t.info or {}):
                        self._rollout_reward_components.append(t.info["reward_components"])
            self._update_info_buffer(infos, dones)
            callback.update_locals({"transitions": transitions, "rollout_buffer": rollout_buffer,
                "dones": dones, "infos": infos, "rewards": np.array([t.reward for t in transitions]),
                "actions": np.array([t.action for t in transitions])})
            if not callback.on_step():
                raise KeyboardInterrupt("Actor collection stopped by callback")

        stats = env.collect_actors(self.policy, n_rollout_steps, consume)
        assert rollout_buffer.full and np.all(rollout_buffer.actor_counts == n_rollout_steps)
        self._last_obs = env._obs
        self._last_episode_starts = last_dones
        with torch.no_grad():
            tensors, _ = self.policy.obs_to_tensor(self._last_obs)
            last_values = self.policy.predict_values(tensors)
        rollout_buffer.compute_returns_and_advantage(last_values, last_dones)
        for name, value in stats.items():
            self.logger.record(f"system/actor_{name}", value)
        callback.on_rollout_end()
        return True

    def train(self):
        started = time.monotonic()
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip = self.clip_range(self._current_progress_remaining)
        value_clip = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None
        history = {name: [] for name in (
            "policy_gradient_loss", "value_loss", "entropy_loss", "placement_loss",
            "placement_label_fraction", "approx_kl", "clip_fraction", "loss",
        )}
        stop = False
        optimizer_steps_before = self.optimizer_step_count()
        optimizer_steps_this_iteration = 0
        for _ in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                advantages = batch.advantages
                if self.normalize_advantage and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.policy.optimizer.zero_grad()
                count = len(advantages)
                micro_size = self.micro_batch_size or count
                known_total = int((batch.placement_targets >= 0).sum().item())
                metrics = {name: 0.0 for name in history}
                for begin in range(0, count, micro_size):
                    part = slice(begin, min(begin + micro_size, count))
                    weight = (part.stop - part.start) / count
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        {key: value[part] for key, value in batch.observations.items()},
                        batch.actions[part].long().flatten(), action_masks=batch.action_masks[part],
                    )
                    log_ratio = log_prob - batch.old_log_prob[part]
                    ratio = log_ratio.exp()
                    policy_loss = -torch.minimum(advantages[part] * ratio,
                                                advantages[part] * ratio.clamp(1 - clip, 1 + clip)).mean()
                    prediction = values.flatten()
                    if value_clip is not None:
                        prediction = batch.old_values[part] + (prediction - batch.old_values[part]).clamp(-value_clip, value_clip)
                    value_loss = F.mse_loss(prediction, batch.returns[part])
                    entropy_loss = -entropy.mean() if entropy is not None else log_prob.mean()
                    known = batch.placement_targets[part] >= 0
                    placement = self.policy.placement_logits
                    auxiliary = (F.cross_entropy(placement[known], batch.placement_targets[part][known], reduction="sum")
                                 / max(known_total, 1) if known.any() else placement.sum() * 0)
                    loss = weight * (policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss) + self.aux_coef * auxiliary
                    loss.backward()
                    self.policy.placement_logits = None
                    for name, value in {
                        "policy_gradient_loss": weight * policy_loss.item(),
                        "value_loss": weight * value_loss.item(),
                        "entropy_loss": weight * entropy_loss.item(), "placement_loss": auxiliary.item(),
                        "placement_label_fraction": weight * known.float().mean().item(),
                        "approx_kl": weight * ((ratio - 1) - log_ratio).mean().detach().item(),
                        "clip_fraction": weight * ((ratio - 1).abs() > clip).float().mean().item(),
                        "loss": loss.item(),
                    }.items():
                        metrics[name] += value
                for name, value in metrics.items():
                    history[name].append(value)
                if self.target_kl is not None and metrics["approx_kl"] > 1.5 * self.target_kl:
                    self.policy.optimizer.zero_grad()
                    stop = True
                    break
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.policy.optimizer.step()
                optimizer_steps_this_iteration += 1
                if getattr(self, "_after_optimizer_step", None) is not None:
                    self._after_optimizer_step(self, {**metrics, "grad_norm": float(grad_norm.item())},
                                               optimizer_steps_before + optimizer_steps_this_iteration)
            self._n_updates += 1
            if stop:
                break
        for name, entries in history.items():
            self.logger.record(f"train/{name}", float(np.mean(entries)) if entries else 0.0)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/optimizer_steps", optimizer_steps_before + optimizer_steps_this_iteration)
        self.logger.record("train/optimizer_steps_this_iteration", optimizer_steps_this_iteration)
        self.logger.record("train/clip_range", clip)
        self.logger.record("train/explained_variance", explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten(),
        ))
        self.last_train_seconds = time.monotonic() - started
        if getattr(self, "_after_update", None) is not None:
            self._after_update(self)
