"""Asynchronous game completion must preserve PPO targets, updates and metrics."""
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
from stable_baselines3.common.callbacks import BaseCallback

from taverngym.learning.actor_sampling import ActorOutput, ActorOpponentVectorEnv, UnifiedActorSampler
from taverngym.learning.buffer import CompactPlacementBuffer
from taverngym.learning.model import ModelConfig
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.opponents import FrozenOpponentPool, OpponentGroup
from taverngym.learning.vector_env import MixedOpponentVectorEnv, NativeVectorEnv


class Rules:
    def __init__(self, slow=False):
        self.slow = slow

    def act(self, obs):
        masks = obs["action_mask"]
        actions = masks.argmax(1)
        actions[masks[:, 2] & (self.slow | (obs["features"][:, 15] == 0))] = 2
        values = obs["features"][:, 0].astype(np.float32)
        return ActorOutput(actions, values, np.zeros(len(actions), np.float32))

    def predict(self, obs, deterministic=False):
        return self.act(obs).actions, None


def test_scattered_actor_buffer_matches_vector_buffer_and_gae():
    env = MixedOpponentVectorEnv(3, seed=12, threads=1, frozen_opponents=2,
        opponent_policy=Rules(True), config={"turn_action_caps": [6]}, reward_shaping={})
    sampler = UnifiedActorSampler(3, Rules(), Rules(True), seed=12, threads=3, frozen_opponents=2,
        config={"turn_action_caps": [6]}, reward_shaping={})
    buffers = [CompactPlacementBuffer(180, env.observation_space, env.action_space,
               n_envs=3, gamma=1, gae_lambda=.995, device="cpu") for _ in range(2)]
    sync, actor = buffers
    obs = env.reset()
    starts = np.ones(3, dtype=bool)
    try:
        for _ in range(180):
            out = Rules().act(obs)
            after, rewards, dones, infos = env.step(out.actions)
            sync.record_endings(dones, infos)
            sync.add(obs, out.actions, rewards, starts, torch.from_numpy(out.values), torch.from_numpy(out.log_probs))
            obs, starts = after, dones
        sampler.collect(180, on_batch=actor.add_actor_batch)
        assert actor.full and np.all(actor.actor_counts == 180)
        last_values = torch.from_numpy(Rules().act(obs).values)
        for buffer in buffers:
            buffer.compute_returns_and_advantage(last_values, starts)
        for name in ("actions", "rewards", "episode_starts", "values", "log_probs", "placement_targets", "advantages", "returns"):
            np.testing.assert_array_equal(getattr(sync, name), getattr(actor, name))
        for want, got in zip(sync._storage, actor._storage):
            np.testing.assert_array_equal(want, got)
        assert (actor.placement_targets >= 0).any() and (actor.rewards < 0).any()
    finally:
        env.close()
        for buffer in buffers:
            buffer.close()


class Capture(BaseCallback):
    def __init__(self):
        super().__init__()
        self.completed = []

    def _on_step(self):
        self.completed.extend(t for t in self.locals["transitions"] if t.done)
        return True


@pytest.mark.parametrize("multiple_versions", [False, True])
def test_actor_ppo_updates_fresh_weights_and_logs_complete_episodes(tmp_path, multiple_versions):
    import copy

    torch.set_num_threads(2)
    reference = NativeVectorEnv(2, seed=4, config={"turn_action_caps": [3]})
    fixed_model = EntityPPO(EntityPolicy, reference, n_steps=8, batch_size=8, device="cpu", seed=4,
        policy_kwargs={"forward_precision": "bf16", "model_config": asdict(
            ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})
    fixed = fixed_model.policy
    text = torch.randn(fixed.network.identity.num_embeddings, 16)
    text[0].zero_()
    fixed.network.set_text_embeddings(text)
    fixed.requires_grad_(False)
    weights = {name: value.clone() for name, value in fixed.state_dict().items()}
    other = copy.deepcopy(fixed) if multiple_versions else None
    opponents = (FrozenOpponentPool((OpponentGroup("bot_v1", fixed, 2), OpponentGroup("bot_v2", other, 3)))
                 if multiple_versions else fixed)
    env = ActorOpponentVectorEnv(2, seed=4, threads=1, opponent_policy=opponents,
                                frozen_opponents=5 if multiple_versions else 2,
                                config={"turn_action_caps": [3]}, reward_shaping={})
    model = EntityPPO(EntityPolicy, env, n_steps=64, batch_size=64, micro_batch_size=16, n_epochs=1,
        rollout_buffer_class=CompactPlacementBuffer, device="cpu", seed=4, policy_kwargs=fixed_model.policy_kwargs)
    model.policy.load_state_dict(fixed.state_dict())
    before = model.policy.network.pointer[-1].weight.detach().clone()
    capture = Capture()
    try:
        model.learn(256, callback=capture, log_interval=None)
        assert model.num_timesteps == 256 and model.optimizer_step_count() == 4
        assert not torch.equal(before, model.policy.network.pointer[-1].weight)
        assert all(torch.equal(value, weights[name]) for name, value in fixed.state_dict().items())
        assert not fixed.optimizer.state and all(p.grad is None for p in fixed.parameters())
        if other is not None:
            assert all(torch.equal(value, weights[name]) for name, value in other.state_dict().items())
            assert not other.optimizer.state and all(p.grad is None for p in other.parameters())
            assert other.network.identity.weight.data_ptr() != model.policy.network.identity.weight.data_ptr()
        assert env.opponent_decisions > 0 and capture.completed
        for transition in capture.completed:
            info = transition.info
            assert info["episode"]["r"] == pytest.approx(info["reward_components"]["total_reward"])
            assert info["reward_components"]["actions_per_game"] == info["episode"]["l"]
            assert 0 <= info["reward_components"]["cap_end_rate"] <= 1
        assert model.logger.name_to_value["system/actor_learner_samples"] == 128
        assert "rollout/cycle_returns" in model.logger.name_to_value
        assert np.isfinite(model.rollout_buffer.advantages).all()
        path = tmp_path / "actor.zip"
        model.save(path)
        restored = EntityPPO.load(path, device="cpu", n_envs=1, n_steps=2, batch_size=2)
        assert restored.optimizer_step_count() == 4 and restored.policy.forward_precision == "bf16"
    finally:
        env.close()
        reference.close()
