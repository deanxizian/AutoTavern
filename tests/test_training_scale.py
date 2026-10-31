"""Batching and storage equivalence checks before committing a large PPO run."""

from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from stable_baselines3.common.callbacks import BaseCallback
from taverngym.learning.buffer import CompactPlacementBuffer, DiskPlacementBuffer, ObservationCodec
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.model import ModelConfig
from taverngym.learning.schedule import LinearLearningRate
from taverngym.learning.vector_env import NativeVectorEnv


class Continue(BaseCallback):
    def _on_step(self):
        return True


def model_for(env, **kwargs):
    config = ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16)
    result = EntityPPO(EntityPolicy, env, policy_kwargs={"model_config": asdict(config)},
                       n_steps=16, batch_size=32, n_epochs=1, seed=4, device="cpu", **kwargs)
    torch.manual_seed(5)
    text = torch.randn(3984, 16)
    text[0].zero_()
    result.policy.network.set_text_embeddings(text)
    return result


def test_native_batch_is_thread_deterministic_and_completes_natural_episodes():
    a, b = NativeVectorEnv(3, seed=66, threads=1), NativeVectorEnv(3, seed=66, threads=3)
    x, y = a.reset(), b.reset()
    completed = 0
    for _ in range(300):
        for name in x:
            np.testing.assert_array_equal(x[name], y[name])
        actions = np.array([np.flatnonzero(mask)[0] for mask in x["action_mask"]])
        x, r1, d1, i1 = a.step(actions)
        y, r2, d2, i2 = b.step(actions)
        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(d1, d2)
        for done, info in zip(d1, i1):
            if done:
                assert 1 <= info["placement"] <= 8 and info["episode"]["l"] > 0
                completed += 1
    assert completed > 0
    a.close()
    b.close()


@pytest.mark.parametrize("buffer_class", [DiskPlacementBuffer, CompactPlacementBuffer])
def test_rollout_storage_matches_memory_and_accumulated_updates(tmp_path, buffer_class):
    torch.set_num_threads(2)
    a = model_for(NativeVectorEnv(2, seed=4))
    b = model_for(NativeVectorEnv(2, seed=4), micro_batch_size=4,
                  rollout_buffer_class=buffer_class,
                  rollout_buffer_kwargs={"storage_root": str(tmp_path)} if buffer_class is DiskPlacementBuffer else {})
    b.policy.load_state_dict(a.policy.state_dict())
    callbacks = []
    for model in (a, b):
        torch.manual_seed(123)
        _, callback = model._setup_learn(32, Continue())
        assert model.collect_rollouts(model.env, callback, model.rollout_buffer, 16)
        callbacks.append(callback)
    left, right = a.rollout_buffer, b.rollout_buffer
    unpacked = (right.observations if buffer_class is DiskPlacementBuffer else
                right.codec.decode(*right._storage))
    for key in left.observations:
        np.testing.assert_array_equal(left.observations[key], unpacked[key])
    for name in ("actions", "rewards", "values", "log_probs", "advantages", "returns", "placement_targets"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    for model in (a, b):
        np.random.seed(111)
        model._current_progress_remaining = 0.5
        model.train()
    for name, value in a.policy.state_dict().items():
        torch.testing.assert_close(value, b.policy.state_dict()[name], rtol=5e-5, atol=2e-6, msg=name)
    # File pages and numerical observation precision remain unchanged after minibatch reads.
    assert unpacked["features"].dtype == np.float32
    assert right.placement_targets.ndim == 2
    b.rollout_buffer.close()
    a.env.close()
    b.env.close()


def test_compact_codec_is_bitwise_lossless_on_real_observations():
    env, codec = NativeVectorEnv(3, seed=88), ObservationCodec()
    obs = env.reset()
    rng = np.random.default_rng(99)
    for _ in range(300):
        restored = codec.decode(*codec.encode(obs))
        np.testing.assert_array_equal(obs["features"].view(np.uint32), restored["features"].view(np.uint32))
        np.testing.assert_array_equal(obs["cards"], restored["cards"])
        np.testing.assert_array_equal(obs["action_mask"], restored["action_mask"])
        actions = [int(rng.choice(np.flatnonzero(mask))) for mask in obs["action_mask"]]
        obs, _, _, _ = env.step(actions)
    assert len(codec.numeric) == 677 and len(codec.binary) == 2550
    assert codec.bytes_per_observation() == 3306
    env.close()


def test_linear_scheduler_endpoints_and_clamping():
    schedule = LinearLearningRate()
    assert schedule(1) == pytest.approx(3e-4)
    assert schedule(0.5) == pytest.approx(1.515e-4)
    assert schedule(0) == pytest.approx(3e-6)
    assert schedule(2) == schedule(1)
    assert schedule(-1) == schedule(0)


def test_evaluation_is_reproducible_and_preserves_training_rng():
    from taverngym.learning.experiment import evaluate

    model = model_for(NativeVectorEnv(2, seed=4))
    state = torch.random.get_rng_state().clone()
    first = evaluate(model, 2, seed=101, n_envs=2, simulator_config={"turn_action_caps": [3]})
    assert torch.equal(state, torch.random.get_rng_state())
    second = evaluate(model, 2, seed=101, n_envs=2, simulator_config={"turn_action_caps": [3]})
    assert first == second and first["games"] == 2
    assert all(1 <= place <= 8 for place in first["placements"])
    model.env.close()


def test_experiment_resume_preserves_completed_iterations_and_optimizer(tmp_path):
    import json
    import subprocess
    import sys

    model = model_for(NativeVectorEnv(2, seed=4))
    initial = tmp_path / "initial.zip"
    model.save(initial)
    config = {
        "name": "resume-test", "iterations": 1, "decisions_per_iteration": 32,
        "n_envs": 2, "n_steps": 16, "batch_size": 16, "micro_batch_size": 4, "n_epochs": 1,
        "learning_rate_start": 6e-6, "learning_rate_end": 5e-6,
        "seed": 4, "env_threads": 1, "checkpoint_every_iterations": 1,
        "evaluation_every_iterations": 1, "evaluation_games": 0,
        "initial_evaluation_games": 0, "final_evaluation_games": 0,
        "evaluation_seed": 101, "final_evaluation_seed": 102,
        "simulator_config": {"turn_action_caps": [3]},
        "tensorboard_train_per_optimizer_step": True,
        "tensorboard_step_unit": "optimizer_step",
        "tensorboard_validation_prefix": "val",
        "tensorboard_baseline_prefix": "baseline",
        "tracking_backend": "tensorboard",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    experiment, run = tmp_path / "experiment", tmp_path / "run"
    base = [sys.executable, "-m", "taverngym.learning.experiment", "--config", str(config_path),
            "--experiment-dir", str(experiment), "--run-dir", str(run), "--device", "cpu"]
    subprocess.run(base + ["--model", str(initial)], check=True, capture_output=True, text=True)
    source = tmp_path / "first-iteration.zip"
    source.write_bytes((run / "latest_model.zip").read_bytes())
    before = EntityPPO.load(source, device="cpu")
    assert before.num_timesteps == 32 and before.policy.optimizer.state
    config["iterations"] = 3
    config_path.write_text(json.dumps(config))
    subprocess.run(base + ["--model", str(source), "--resume"], check=True, capture_output=True, text=True)
    after = EntityPPO.load(run / "final_model.zip", device="cpu")
    assert after.num_timesteps == 96 and after._n_updates == 3
    records = [json.loads(line) for line in (experiment / "metrics.jsonl").read_text().splitlines()]
    assert [row["iteration"] for row in records] == [1, 2, 3]
    assert [row["train/optimizer_steps"] for row in records] == [2, 4, 6]
    assert [row["train/optimizer_steps_this_iteration"] for row in records] == [2, 2, 2]
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(experiment)).Reload()
    assert [event.step for event in events.Scalars("train/loss")] == [1, 2]
    assert records[-1]["train/learning_rate"] == pytest.approx(5e-6)
    old_step = next(iter(before.policy.optimizer.state.values()))["step"]
    new_step = next(iter(after.policy.optimizer.state.values()))["step"]
    assert new_step > old_step
    resume = json.loads(next((experiment / "resume-history").glob("*.json")).read_text())
    assert resume["completed_iterations"] == 1 and resume["target_iterations"] == 3
    assert (experiment / "progress.csv").exists() and (experiment / "sessions").exists()
    model.env.close()
