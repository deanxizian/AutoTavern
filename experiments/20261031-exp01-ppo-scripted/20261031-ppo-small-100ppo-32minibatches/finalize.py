"""Verify completion, evaluate the preselected best and apply best-only retention."""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from taverngym.learning.experiment import evaluate, now, write_json
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.schedule import LinearLearningRate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    folder = Path(__file__).resolve().parent
    repository = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
    models = repository / "runs" / folder.name
    config = json.loads((folder / "config.json").read_text())
    resolved = json.loads((folder / "resolved-config.json").read_text())
    best = models / resolved.get("best_model_filename", "best_model.zip")
    progress = json.loads((folder / "progress.json").read_text())
    assert progress["status"] == "completed" and progress["completed_iterations"] == config["iterations"]
    total = config["iterations"] * config["decisions_per_iteration"]
    metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == config["iterations"]
    schedule = LinearLearningRate(config["learning_rate_start"], config["learning_rate_end"])
    for index, row in enumerate(metrics, 1):
        assert row["iteration"] == index and row["decisions"] == index * config["decisions_per_iteration"]
        assert math.isclose(row["train/learning_rate"], schedule(1 - index / config["iterations"]), rel_tol=1e-10)
        assert math.isfinite(row["train/loss"])
    final = EntityPPO.load(models / "final_model.zip", device="cpu")
    steps = final.optimizer_step_count()
    assert final.num_timesteps == total
    assert sum(int(row["train/optimizer_steps_this_iteration"]) for row in metrics) == steps
    assert steps <= config["derived_budget"]["total_optimizer_updates_max"]
    assert all(torch.isfinite(parameter).all() for parameter in final.policy.parameters())
    final_hash = digest(models / "final_model.zip")
    final.rollout_buffer.close()
    del final
    evaluations = [(int(path.stem.split("-")[-1]), json.loads(path.read_text()))
                   for path in sorted(folder.glob("evaluation-[0-9][0-9][0-9][0-9].json"))]
    selected_iteration, selected = min(evaluations, key=lambda item: item[1]["mean_placement"])
    source = repository / "runs/20261031-ppo-small-init/initial_model.zip"
    metadata = json.loads((folder / "experiment.json").read_text())
    assert digest(source) == metadata["source_sha256"]
    heldout = {"final": json.loads((folder / "evaluation-final-heldout.json").read_text())}
    best_steps = None
    for label, path in (("best", best), ("initial", source)):
        model = EntityPPO.load(path, device="cuda")
        if label == "best":
            assert model.num_timesteps == selected_iteration * config["decisions_per_iteration"]
            best_steps = model.optimizer_step_count()
            assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
        result = evaluate(model, config["final_evaluation_games"], config["final_evaluation_seed"],
                          simulator_config=config.get("simulator_config"))
        result.update(model=label, model_timesteps=model.num_timesteps,
                      optimizer_steps=model.optimizer_step_count(), used_for_selection=False)
        write_json(folder / f"evaluation-{label}-heldout.json", result)
        heldout[label] = result
        print(label, {key: result[key] for key in ("mean_placement", "first_rate", "top4_rate")}, flush=True)
        close = getattr(model.rollout_buffer, "close", None)
        if close:
            close()
        del model
    with SummaryWriter(str(folder), filename_suffix=".heldout") as writer:
        for label, result in heldout.items():
            step = {"final": steps, "best": best_steps, "initial": 0}[label]
            for name in ("mean_placement", "first_rate", "top4_rate"):
                writer.add_scalar(f"test/{label}_{name}", result[name], step)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(folder), size_guidance={"scalars": 0}).Reload()
    assert len(events.Scalars("train/loss")) == steps
    assert len(events.Scalars("val/mean_placement")) == 4
    assert len(events.Scalars("baseline/mean_placement")) == 1
    verification = {"status": "passed", "verified_at": now(), "completed_ppo_iterations": config["iterations"],
                    "fresh_samples": total, "actual_adam_updates": steps,
                    "max_adam_updates": config["derived_budget"]["total_optimizer_updates_max"],
                    "train_tensorboard_points": steps, "validation_tensorboard_points": 4,
                    "baseline_tensorboard_points": 1, "tensorboard_x_axis": "actual Adam optimizer steps",
                    "linear_schedule_verified": True, "source_initialization_unchanged": True,
                    "final_model_sha256_before_cleanup": final_hash,
                    "best_ppo_iteration": selected_iteration, "best_optimizer_steps": best_steps}
    write_json(folder / "verification.json", verification)
    candidates = [models / name for name in ("starting_model.zip", "latest_model.zip", "final_model.zip")]
    candidates += [models / "checkpoints" / f"iteration-{i:04d}.zip"
                   for i in range(config["checkpoint_every_iterations"], config["iterations"] + 1,
                                  config["checkpoint_every_iterations"])]
    assert models.resolve().parent == (repository / "runs").resolve()
    assert set(models.rglob("*.zip")).issubset({best, *candidates})
    assert best.is_file() and not best.is_symlink()
    kept = {"path": str(best.relative_to(repository)), "sha256": digest(best), "bytes": best.stat().st_size,
            "ppo_iteration": selected_iteration, "optimizer_steps": best_steps}
    removed = []
    for path in candidates:
        if path.exists():
            assert not path.is_symlink()
            removed.append({"path": str(path.relative_to(repository)), "sha256": digest(path), "bytes": path.stat().st_size})
    retention = {"status": "planned", "kept_model": kept, "removed_models": removed,
                 "selection_mean_placement": selected["mean_placement"], "heldout_used_for_selection": False}
    write_json(folder / "retention.json", retention)
    for path in candidates:
        if path.exists():
            path.unlink()
    if (models / "checkpoints").exists() and not any((models / "checkpoints").iterdir()):
        (models / "checkpoints").rmdir()
    assert digest(best) == kept["sha256"] and list(models.rglob("*.zip")) == [best]
    retention.update(status="completed", completed_at=now(), reclaimed_bytes=sum(row["bytes"] for row in removed))
    write_json(folder / "retention.json", retention)
    results = {**verification, "elapsed_training_seconds": progress["elapsed_seconds"],
               "sampling_seconds": sum(row["rollout_seconds"] for row in metrics),
               "update_seconds": sum(row["update_seconds"] for row in metrics),
               "selection": {"ppo_iteration": selected_iteration, "mean_placement": selected["mean_placement"]},
               "heldout": heldout, "retained_model": kept}
    write_json(folder / "results.json", results)
    metadata.update(status="completed", completed_at=now(), completed_ppo_iterations=config["iterations"],
                    completed_samples=total, actual_adam_updates=steps, retained_model=kept,
                    results="results.json", retention="retention.json", verification="verification.json")
    write_json(folder / "experiment.json", metadata)
    progress.update(latest_model=str(best.relative_to(repository)), retained_ppo_iteration=selected_iteration,
                    retention_policy="best_only", phase="completed")
    write_json(folder / "progress.json", progress)
    print("Training, held-out verification and best-only retention completed.", flush=True)


if __name__ == "__main__":
    main()
