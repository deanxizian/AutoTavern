"""Publish the action-loop diagnostic scalars without uploading model files."""

import json
from pathlib import Path

from taverngym.learning.tracking import WandbTracker


folder = Path(__file__).resolve().parent
results = json.loads((folder / "results.json").read_text())
tracker = WandbTracker(folder, {"name": folder.name, "job": "action-loop-diagnostic",
                               "model_sha256": results["model_sha256"], "seed": results["seed"],
                               "simulator_config": results["simulator_config"]}, kind="diagnostic")
try:
    metrics = {}
    for mode, result in results["modes"].items():
        for name in ("mean_placement", "first_rate", "top4_rate", "games_with_move_returns",
                     "games_with_freeze_returns", "games_with_cap_end", "max_reversible_streak"):
            metrics[f"diagnostic/{mode}_{name}"] = result[name]
        for name, value in result["totals"].items():
            metrics[f"diagnostic/{mode}_{name}"] = value
    tracker.log(metrics, 3200)
    metadata = json.loads((folder / "experiment.json").read_text())
    metadata["wandb_url"] = tracker.run.url
    (folder / "experiment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(tracker.run.url, flush=True)
finally:
    tracker.finish()
