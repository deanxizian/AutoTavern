"""Logging must preserve individual updates, evaluation points and live files."""

import json
import subprocess
import sys
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from taverngym.learning.tracking import metric_name


def test_flat_metric_names_share_baseline_and_validation():
    assert metric_name("baseline/mean_placement") == "val/mean_placement"
    assert metric_name("val/mean_placement") == "val/mean_placement"
    assert metric_name("rollout/mean_placement") == "train/mean_placement"
    assert metric_name("ppo_summary/approx_kl") == "train/ppo_approx_kl"
    assert metric_name("train/rollout/top4_rate") == "train/rollout_top4_rate"


def test_live_bridge_reads_both_active_event_files_and_resumes(tmp_path):
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter
    from taverngym.learning.migrate_wandb import EventFiles, journal_keys

    first = SummaryWriter(str(tmp_path), filename_suffix=".summary")
    second = SummaryWriter(str(tmp_path), filename_suffix=".adam")
    first.add_scalar("baseline/mean_placement", 8.0, 0)
    second.add_scalar("train/loss", .5, 1)
    first.flush()
    second.flush()
    seen, reader = set(), EventFiles(tmp_path)
    batches, counts = reader.reload(seen)
    assert [step for step, _ in batches] == [0, 1]
    assert counts["train/loss"] == 1
    # The older summary file is still being written after the newer Adam file.
    first.add_scalar("val/mean_placement", 4.0, 2)
    second.add_scalar("train/loss", .4, 2)
    first.flush()
    second.flush()
    batches, _ = reader.reload(seen)
    assert batches == [(2, {"val/mean_placement": 4.0, "train/loss": pytest.approx(.4)})]
    journal = tmp_path / "wandb-metrics.jsonl"
    journal.write_text('\n'.join(json.dumps({name: 1, "optimizer_step": step}) for name, step in seen) + '\n')
    recovered, _ = EventFiles(tmp_path).reload(journal_keys(journal))
    assert recovered == []
    first.close()
    second.close()


def test_native_wandb_records_each_adam_and_step_zero_without_tensorboard(tmp_path):
    pytest.importorskip("wandb")
    from taverngym.learning.model import ModelConfig
    from taverngym.learning.policy import EntityPolicy
    from taverngym.learning.ppo import EntityPPO
    from taverngym.learning.vector_env import NativeVectorEnv

    torch.set_num_threads(2)
    env = NativeVectorEnv(2, seed=4)
    model = EntityPPO(EntityPolicy, env, n_steps=16, batch_size=16, n_epochs=1,
                      policy_kwargs={"model_config": asdict(ModelConfig(
                          d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))},
                      seed=4, device="cpu")
    text = torch.randn(model.policy.network.identity.num_embeddings, 16)
    text[0].zero_()
    model.policy.network.set_text_embeddings(text)
    initial = tmp_path / "initial.zip"
    model.save(initial)
    config = {
        "name": "offline-logging-test", "iterations": 2, "decisions_per_iteration": 32,
        "n_envs": 2, "n_steps": 16, "batch_size": 16, "micro_batch_size": 4, "n_epochs": 1,
        "learning_rate_start": 3e-4, "learning_rate_end": 3e-6,
        "seed": 4, "env_threads": 1, "checkpoint_every_iterations": 1,
        "evaluation_every_iterations": 1, "evaluation_games": 1,
        "initial_evaluation_games": 1, "final_evaluation_games": 0,
        "evaluation_seed": 101, "final_evaluation_seed": 102,
        "simulator_config": {"turn_action_caps": [3]},
        "tracking": {"mode": "offline"},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    folder, runs = tmp_path / "experiment", tmp_path / "models"
    result = subprocess.run([sys.executable, "-m", "taverngym.learning.experiment",
                             "--config", str(config_path), "--model", str(initial),
                             "--experiment-dir", str(folder), "--run-dir", str(runs),
                             "--device", "cpu"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in (folder / "wandb-metrics.jsonl").read_text().splitlines()]
    assert [r["optimizer_step"] for r in rows if "train/loss" in r] == [1, 2, 3, 4]
    assert [r["optimizer_step"] for r in rows if "val/mean_placement" in r] == [0, 2, 4]
    assert len([r for r in rows if "train/ppo_loss" in r]) == 2
    assert all(name.count("/") <= 1 for row in rows for name in row)
    assert not list(folder.rglob("events.out.tfevents.*"))
    assert not (folder / "tensorboard").exists()
    restored = EntityPPO.load(runs / "final_model.zip", device="cpu")
    assert restored.optimizer_step_count() == 4
    assert getattr(restored, "_after_optimizer_step", None) is None
    model.env.close()
