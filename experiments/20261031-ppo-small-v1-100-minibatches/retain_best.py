"""Keep only the preselected best checkpoint for this completed experiment."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch

from taverngym.learning.ppo import EntityPPO


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    folder = Path(__file__).resolve().parent
    repository = folder.parent.parent
    models = repository / "runs" / folder.name
    assert models.resolve().parent == (repository / "runs").resolve()
    progress = json.loads((folder / "progress.json").read_text())
    verification = json.loads((folder / "verification.json").read_text())
    comparison = json.loads((folder / "heldout-comparison.json").read_text())
    assert progress["status"] == "completed" and progress["completed_iterations"] == 100
    assert verification["status"] == "passed" and verification["adam_updates_verified_on_shared_identity"] == 100
    best = models / "best_model.zip"
    assert best.is_file() and not best.is_symlink()
    model = EntityPPO.load(best, device="cpu")
    selected = comparison["selection"]["iteration"]
    assert model.num_timesteps == selected * 8192
    assert int(model.policy.optimizer.state[model.policy.network.identity.weight]["step"].item()) == selected
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    candidates = [models / name for name in ("starting_model.zip", "latest_model.zip", "final_model.zip")]
    candidates += [models / "checkpoints" / f"iteration-{step:04d}.zip" for step in (25, 50, 75, 100)]
    assert set(models.rglob("*.zip")).issubset({best, *candidates}), "Unexpected models: inspect before deleting"
    manifest_path = folder / "retention.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["status"] == "completed":
            assert digest(best) == previous["kept_model"]["sha256"]
            assert list(models.rglob("*.zip")) == [best]
            print("Retention already completed; best checkpoint hash verified.")
            return
    kept = {"path": str(best.relative_to(repository)), "sha256": digest(best),
            "bytes": best.stat().st_size, "training_updates": selected,
            "samples": model.num_timesteps}
    removed = []
    for path in candidates:
        if path.exists():
            assert not path.is_symlink()
            removed.append({"path": str(path.relative_to(repository)),
                            "sha256": digest(path), "bytes": path.stat().st_size})
    manifest = {"status": "planned", "requested_policy": "Only keep best version after evaluation, scoped to this experiment",
                "selection": comparison["selection"], "best_heldout_mean_placement": comparison["best"]["mean_placement"],
                "completed_training_updates": 100, "kept_model": kept, "removed_models": removed,
                "source_initialization_and_other_runs_preserved": True, "experiment_metrics_and_tensorboard_preserved": True}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for path in candidates:
        if path.exists():
            path.unlink()
    checkpoints = models / "checkpoints"
    if checkpoints.exists() and not any(checkpoints.iterdir()):
        checkpoints.rmdir()
    assert digest(best) == kept["sha256"] and list(models.rglob("*.zip")) == [best]
    manifest.update(status="completed", completed_at=datetime.now().astimezone().isoformat(),
                    removed_count=len(removed), reclaimed_bytes=sum(item["bytes"] for item in removed))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    progress.update(latest_model=str(best.relative_to(repository)), retained_model_updates=selected,
                    retention_policy="best_only")
    (folder / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    print(json.dumps({"kept": kept, "removed_count": len(removed),
                      "reclaimed_mib": manifest["reclaimed_bytes"] / 1024**2}, indent=2))


if __name__ == "__main__":
    main()
