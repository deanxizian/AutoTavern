"""Publish benchmark scalar results; model weights are never uploaded."""

import json
from pathlib import Path

from taverngym.learning.tracking import WandbTracker


folder = Path(__file__).resolve().parent
results = json.loads((folder / "results.json").read_text())
metadata = json.loads((folder / "experiment.json").read_text())
tracker = WandbTracker(folder, {"name": folder.name, "job": "TavernBench eval",
                               "protocol": json.loads((folder / "protocol.json").read_text()),
                               "models": metadata["models"]}, kind="benchmark")
try:
    for model, step in (("ppo-init", 0), ("ppo-100", 3200)):
        metrics = {}
        for suffix in ("policy", "combat256"):
            system = model + "-" + suffix
            row = results["systems"][system]
            prefix = "benchmark/" + system.replace("-", "_") + "_"
            metrics[prefix + "play"] = row["play"]
            metrics[prefix + "strategy_23_scenario_mean"] = row["strategy_23_scenario_mean"]
            metrics[prefix + "coverage"] = row["coverage"]["play_ok"] / 288
            for key in ("positioning", "ordering", "targets", "selection", "strategy", "economy"):
                metrics[prefix + key] = row[key]
        tracker.log(metrics, step)
    tracker.run.summary.update({"evaluated_tasks_per_system": 368, "model_weights_uploaded": False,
                                "trained_policy_play": results["systems"]["ppo-100-policy"]["play"],
                                "trained_augmented_play": results["systems"]["ppo-100-combat256"]["play"]})
    metadata["wandb_url"] = tracker.run.url
    (folder / "experiment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("W&B benchmark:", tracker.run.url, flush=True)
finally:
    tracker.finish()
