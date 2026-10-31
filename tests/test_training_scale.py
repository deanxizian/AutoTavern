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
from taverngym.learning.schedule import LinearLearningRate, WarmupLinearLearningRate, learning_rate_schedule
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


def test_warmup_schedule_has_one_peak_and_roundtrips():
    import pickle

    schedule = learning_rate_schedule({"learning_rate_schedule": "warmup_linear",
        "learning_rate_start": 3e-6, "learning_rate_peak": 1e-5,
        "learning_rate_end": 5e-6, "learning_rate_warmup_fraction": .05})
    values = np.array([schedule(1 - i / 100) for i in range(101)])
    assert values[0] == pytest.approx(3e-6)
    assert values[1] == pytest.approx(4.4e-6)
    assert values[5] == pytest.approx(1e-5)
    assert values[-1] == pytest.approx(5e-6)
    assert [schedule(1 - i / 200) for i in (1, 10, 200)] == pytest.approx([3.7e-6, 1e-5, 5e-6])
    assert np.all(np.diff(values[:6]) > 0) and np.all(np.diff(values[5:]) < 0)
    restored = pickle.loads(pickle.dumps(schedule))
    assert [restored(p) for p in (2, 1, .99, .95, .5, 0, -1)] == [
        schedule(p) for p in (2, 1, .99, .95, .5, 0, -1)]


@pytest.mark.parametrize("settings", [
    {"warmup_fraction": 0}, {"warmup_fraction": 1}, {"peak": 1e-6},
    {"initial": -1e-6}, {"final": float("nan")},
])
def test_invalid_warmup_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        WarmupLinearLearningRate(**settings)


def test_continuation_cli_preserves_adam_and_batches_two_versions(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys

    torch.set_num_threads(2)
    model = model_for(NativeVectorEnv(2, seed=4))
    model.learn(32)
    model.experiment_config = {"decisions_per_iteration": 32}
    source = tmp_path / "source.zip"
    model.save(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source_steps = model.optimizer_total_step_count()
    config = {
        "name": "continue-test", "iterations": 4, "decisions_per_iteration": 32,
        "n_envs": 2, "n_steps": 16, "batch_size": 16, "micro_batch_size": 4, "n_epochs": 1,
        "learning_rate_schedule": "warmup_linear", "learning_rate_start": 3e-6,
        "learning_rate_peak": 1e-5, "learning_rate_end": 5e-6, "learning_rate_warmup_fraction": .5,
        "preserve_optimizer_state": True, "initial_model_sha256": source_hash,
        "sampling_driver": "unified_actor", "frozen_opponents": 5,
        "opponent_pool": [{"name": name, "count": count, "model": str(source), "sha256": source_hash}
                          for name, count in (("bot_v1", 2), ("bot_v2", 3))],
        "seed": 4, "env_threads": 1, "checkpoint_every_iterations": 1,
        "evaluation_every_iterations": 1, "evaluation_games": 0,
        "initial_evaluation_games": 0, "final_evaluation_games": 0,
        "evaluation_seed": 101, "final_evaluation_seed": 102,
        "simulator_config": {"turn_action_caps": [3]}, "tracking_backend": "none",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    folder, run = tmp_path / "experiment", tmp_path / "run"
    command = [sys.executable, "-m", "taverngym.learning.experiment", "--config", str(config_path),
               "--model", str(source), "--experiment-dir", str(folder), "--run-dir", str(run),
               "--device", "cpu", "--continue-stage"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    saved = EntityPPO.load(run / "final_model.zip", device="cpu", n_envs=1, n_steps=2, batch_size=2)
    assert saved.num_timesteps == 128 and saved.optimizer_step_count() == 8
    assert saved.optimizer_total_step_count() == source_steps + 8
    baseline = EntityPPO.load(run / "starting_model.zip", device="cpu", n_envs=1, n_steps=2, batch_size=2)
    assert baseline.optimizer_step_count() == baseline.num_timesteps == 0
    for key, state in model.policy.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, baseline.policy.optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert [r["train/optimizer_steps"] for r in records] == [2, 4, 6, 8]
    assert [r["train/learning_rate"] for r in records] == pytest.approx([6.5e-6, 1e-5, 7.5e-6, 5e-6])
    assert all(r["system/actor_bot_v1_forward_calls"] > 0 and r["system/actor_bot_v2_forward_calls"] > 0
               for r in records)
    resolved = json.loads((folder / "resolved-config.json").read_text())
    assert resolved["stage_source_adam_total_steps"] == source_steps
    assert resolved["preserve_optimizer_state"] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    model.env.close()


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


@pytest.mark.parametrize("continued_stage", [False, True])
def test_experiment_resume_preserves_completed_iterations_and_optimizer(tmp_path, continued_stage):
    import json
    import subprocess
    import sys

    model = model_for(NativeVectorEnv(2, seed=4))
    if continued_stage:
        model.learn(32)
    source_timesteps = model.num_timesteps
    source_adam_steps = model.optimizer_total_step_count()
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
    first = base + ["--model", str(initial)] + (["--continue-stage"] if continued_stage else [])
    subprocess.run(first, check=True, capture_output=True, text=True)
    original_best_name = json.loads((experiment / "resolved-config.json").read_text())["best_model_filename"]
    source = tmp_path / "first-iteration.zip"
    source.write_bytes((run / "latest_model.zip").read_bytes())
    before = EntityPPO.load(source, device="cpu")
    assert before.num_timesteps == 32 and before.policy.optimizer.state
    config["iterations"] = 3
    config_path.write_text(json.dumps(config))
    subprocess.run(base + ["--model", str(source), "--resume"], check=True, capture_output=True, text=True)
    assert json.loads((experiment / "resolved-config.json").read_text())["best_model_filename"] == original_best_name
    if continued_stage:
        resolved = json.loads((experiment / "resolved-config.json").read_text())
        assert resolved["stage_source_adam_total_steps"] == source_adam_steps
        assert resolved["warm_start_source_timesteps"] == source_timesteps
        assert resolved["warm_start_source_optimizer_steps"] == source_adam_steps
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
