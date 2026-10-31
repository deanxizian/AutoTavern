"""Evaluate initial/best/final against the fixed roster, then retain stage best."""

import hashlib
import json
from pathlib import Path

import torch

from taverngym.learning.experiment import evaluate_suite, load_frozen_opponents, now, write_json
from taverngym.learning.ppo import EntityPPO, PlacementBuffer
from taverngym.learning.schedule import learning_rate_schedule
from taverngym.learning.tracking import WandbTracker


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    folder = Path(__file__).resolve().parent
    root = next(p for p in folder.parents if (p / "pyproject.toml").is_file())
    models = root / "runs" / folder.name
    config = json.loads((folder / "resolved-config.json").read_text())
    progress = json.loads((folder / "progress.json").read_text())
    assert progress["status"] == "completed" and progress["completed_iterations"] == config["iterations"]
    source = root / config["initial_model"]
    assert digest(source) == config["initial_model_sha256"]
    for spec in config["opponent_pool"]:
        assert digest(root / spec["model"]) == spec["sha256"]
    best = models / config["best_model_filename"]
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == config["iterations"]
    schedule = learning_rate_schedule(config)
    for i, row in enumerate(rows, 1):
        assert row["iteration"] == i and row["decisions"] == i * config["decisions_per_iteration"]
        assert abs(row["train/learning_rate"] - schedule(1 - i / config["iterations"])) < 1e-15
    steps = int(rows[-1]["train/optimizer_steps"])
    assert steps == sum(int(row["train/optimizer_steps_this_iteration"]) for row in rows)
    evaluations = [(int(p.stem.split("-")[-1]), json.loads(p.read_text()))
                   for p in sorted(folder.glob("evaluation-[0-9][0-9][0-9][0-9].json"))]
    assert len(evaluations) == 5
    selected_iteration, _ = min(evaluations, key=lambda item: item[1]["mean_placement"])
    opponents = load_frozen_opponents(config, "cuda")
    heldout = {"final": json.loads((folder / "evaluation-final-heldout.json").read_text())}
    for name, path in (("initial", source), ("best", best)):
        model = EntityPPO.load(path, device="cuda", n_envs=1, n_steps=2, batch_size=2,
                               rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        stage_steps = 0
        if name == "best":
            stage_steps = model.optimizer_step_count()
            assert model.num_timesteps == selected_iteration * config["decisions_per_iteration"]
            assert model.optimizer_total_step_count() == config["stage_source_adam_total_steps"] + stage_steps
        result = (heldout["final"].copy() if name == "best" and selected_iteration == config["iterations"]
                  else evaluate_suite(model, config["final_evaluation_games"], config["final_evaluation_seed"],
                                      config, opponents))
        result.update(model=name, stage_optimizer_steps=stage_steps, used_for_selection=False)
        write_json(folder / f"evaluation-{name}-heldout.json", result)
        heldout[name] = result
        del model
    tracker = WandbTracker(folder, config)
    exit_code = 1
    try:
        for name, result in heldout.items():
            values = {f"test/{name}_{condition}_{key}": row[key]
                      for condition, row in result["conditions"].items()
                      for key in ("mean_placement", "first_rate", "top4_rate", "cycle_returns",
                                  "cap_end_rate", "actions_per_game", "total_reward")}
            tracker.log(values, {"initial": 0, "best": heldout["best"]["stage_optimizer_steps"], "final": steps}[name])
        candidates = [p for p in models.rglob("*.zip") if p != best]
        assert models.resolve().parent == (root / "runs").resolve()
        assert best.is_file() and not best.is_symlink()
        assert all(not p.is_symlink() and p.parent in (models, models / "checkpoints") for p in candidates)
        kept = {"path": str(best.relative_to(root)), "sha256": digest(best), "bytes": best.stat().st_size,
                "ppo_iteration": selected_iteration, "stage_optimizer_steps": heldout["best"]["stage_optimizer_steps"]}
        removed = [{"path": str(p.relative_to(root)), "sha256": digest(p), "bytes": p.stat().st_size}
                   for p in candidates]
        write_json(folder / "retention.json", {"status": "planned", "kept_model": kept, "removed_models": removed})
        for p in candidates:
            p.unlink()
        if (models / "checkpoints").exists() and not any((models / "checkpoints").iterdir()):
            (models / "checkpoints").rmdir()
        assert list(models.rglob("*.zip")) == [best]
        assert digest(source) == config["initial_model_sha256"]
        write_json(folder / "retention.json", {"status": "completed", "completed_at": now(),
                   "kept_model": kept, "removed_models": removed, "heldout_used_for_selection": False})
        write_json(folder / "results.json", {"status": "completed", "completed_at": now(),
                   "fresh_samples": config["iterations"] * config["decisions_per_iteration"],
                   "actual_adam_updates": steps, "retained_model": kept, "heldout": heldout,
                   "initial_model_unchanged": True, "weights_uploaded": False})
        metadata = json.loads((folder / "experiment.json").read_text())
        metadata.update(status="completed", completed_at=now(), results="results.json", retained_model=kept)
        write_json(folder / "experiment.json", metadata)
        progress.update(phase="completed", status="completed", updated_at=now(), latest_model=str(best))
        write_json(folder / "progress.json", progress)
        exit_code = 0
    finally:
        tracker.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
