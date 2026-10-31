"""Verify stage completion, compare held-out policies, benchmark, retain best only."""

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

from taverngym.learning.experiment import evaluate_suite, load_frozen_policy, now, write_json
from taverngym.learning.ppo import EntityPPO, PlacementBuffer
from taverngym.learning.schedule import LinearLearningRate
from taverngym.learning.tracking import WandbTracker


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    folder = Path(__file__).resolve().parent
    root = next(p for p in folder.parents if (p / "pyproject.toml").exists())
    models = root / "runs" / folder.name
    config = json.loads((folder / "resolved-config.json").read_text())
    progress = json.loads((folder / "progress.json").read_text())
    assert progress["status"] == "completed" and progress["completed_iterations"] == config["iterations"]
    source = root / config["initial_model"]
    assert digest(source) == config["opponent_model_sha256"]
    best = models / config["best_model_filename"]
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == config["iterations"]
    schedule = LinearLearningRate(config["learning_rate_start"], config["learning_rate_end"])
    for i, row in enumerate(rows, 1):
        assert row["iteration"] == i and row["decisions"] == i * config["decisions_per_iteration"]
        assert math.isclose(row["train/learning_rate"], schedule(1 - i / config["iterations"]), rel_tol=1e-9)
        assert math.isfinite(row["train/loss"])
    steps = int(rows[-1]["train/optimizer_steps"])
    assert steps == sum(int(row["train/optimizer_steps_this_iteration"]) for row in rows)
    evaluations = [(int(p.stem.split("-")[-1]), json.loads(p.read_text()))
                   for p in sorted(folder.glob("evaluation-[0-9][0-9][0-9][0-9].json"))]
    assert len(evaluations) == 5
    selected_iteration, selected = min(evaluations, key=lambda item: item[1]["mean_placement"])
    journal = [json.loads(line) for line in (folder / "wandb-metrics.jsonl").read_text().splitlines()]
    assert len([row for row in journal if "train/loss" in row]) == steps
    assert len([row for row in journal if "val/mean_placement" in row]) == 5
    assert all(name.count("/") <= 1 for row in journal for name in row)
    opponent = load_frozen_policy(source, "cuda")
    heldout = {"final": json.loads((folder / "evaluation-final-heldout.json").read_text())}
    best_steps = None
    for name, path in (("initial", source), ("best", best)):
        model = EntityPPO.load(path, device="cuda", n_envs=1, n_steps=2, batch_size=2,
                               rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        if name == "best":
            best_steps = model.optimizer_step_count()
            assert model.num_timesteps == selected_iteration * config["decisions_per_iteration"]
        result = (heldout["final"].copy() if name == "best" and selected_iteration == config["iterations"]
                  else evaluate_suite(model, config["final_evaluation_games"], config["final_evaluation_seed"], config, opponent))
        result.update(model=name, stage_optimizer_steps=best_steps if name == "best" else 0,
                      used_for_selection=False)
        write_json(folder / f"evaluation-{name}-heldout.json", result)
        heldout[name] = result
        del model
    tracker = WandbTracker(folder, config)
    try:
        for label, result in heldout.items():
            values = {}
            for condition, metrics in result["conditions"].items():
                for key in ("mean_placement", "first_rate", "top4_rate", "cycle_returns", "cap_end_rate",
                            "operation_penalty", "cycle_penalty", "actions_per_game", "total_reward"):
                    values[f"test/{label}_{condition}_{key}"] = metrics[key]
            tracker.log(values, {"initial": 0, "best": best_steps, "final": steps}[label])
        benchmark = folder.parent / "20261031-ppo-mixed-2v1-bf16-actor-tavernbench"
        subprocess.run([sys.executable, "-u", str(benchmark / "run.py")], cwd=root, check=True)
        bench = json.loads((benchmark / "results.json").read_text())
        tracker.log({f"test/tavernbench_{name.replace('-', '_')}_play": row["play"]
                     for name, row in bench["systems"].items()}, steps)
        candidates = [models / name for name in ("starting_model.zip", "latest_model.zip", "final_model.zip",
                                                 "failed_model.zip", "interrupted_model.zip")]
        candidates += sorted((models / "checkpoints").glob("iteration-*.zip"))
        assert models.resolve().parent == (root / "runs").resolve()
        assert not best.is_symlink() and best.is_file()
        assert set(models.rglob("*.zip")).issubset({best, *candidates})
        kept = {"path": str(best.relative_to(root)), "sha256": digest(best), "bytes": best.stat().st_size,
                "ppo_iteration": selected_iteration, "optimizer_steps": best_steps}
        removed = [{"path": str(p.relative_to(root)), "sha256": digest(p), "bytes": p.stat().st_size}
                   for p in candidates if p.exists()]
        assert all(not p.is_symlink() for p in candidates)
        write_json(folder / "retention.json", {"status": "planned", "kept_model": kept, "removed_models": removed})
        for p in candidates:
            if p.exists():
                p.unlink()
        if (models / "checkpoints").exists() and not any((models / "checkpoints").iterdir()):
            (models / "checkpoints").rmdir()
        assert list(models.rglob("*.zip")) == [best] and digest(source) == config["opponent_model_sha256"]
        write_json(folder / "retention.json", {"status": "completed", "completed_at": now(),
                   "kept_model": kept, "removed_models": removed, "heldout_used_for_selection": False})
        results = {"status": "completed", "completed_at": now(), "iterations": config["iterations"],
                   "fresh_samples": config["total_decisions"], "actual_adam_updates": steps,
                   "retained_model": kept, "heldout": heldout,
                   "sampling_seconds": sum(r["rollout_seconds"] for r in rows),
                   "update_seconds": sum(r["update_seconds"] for r in rows),
                   "source_model_unchanged": True, "weights_uploaded": False}
        write_json(folder / "results.json", results)
        metadata = json.loads((folder / "experiment.json").read_text())
        metadata.update(status="completed", completed_at=now(), retained_model=kept,
                        results="results.json", retention="retention.json", benchmark=str(benchmark.relative_to(root)),
                        exit_code=0, exited_at=now())
        write_json(folder / "experiment.json", metadata)
        progress.update(status="completed", phase="completed", retained_model=kept,
                        latest_model=str(best), updated_at=now())
        write_json(folder / "progress.json", progress)
    finally:
        tracker.finish()
    print("Held-out evaluation, best-only retention and TavernBench completed.", flush=True)


if __name__ == "__main__":
    main()
