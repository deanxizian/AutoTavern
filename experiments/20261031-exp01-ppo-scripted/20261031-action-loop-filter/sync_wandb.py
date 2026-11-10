"""Publish before/after scalars; do not upload weights or gameplay snapshots."""

import json
from pathlib import Path

from taverngym.learning.tracking import WandbTracker


folder = Path(__file__).resolve().parent
after = json.loads((folder / "results.json").read_text())
before = json.loads((folder / after["baseline"]).read_text())
tracker = WandbTracker(folder, {"name": folder.name, "job": "action-loop-filter",
                               "filter": after["filter"], "model_sha256": after["model_sha256"],
                               "seed": after["seed"]}, kind="diagnostic")
try:
    values = {}
    for label, results in (("before", before), ("after", after)):
        for mode, result in results["modes"].items():
            for name in ("mean_placement", "first_rate", "top4_rate", "games_with_move_returns",
                         "games_with_freeze_returns", "max_reversible_streak"):
                values[f"filter/{mode}_{label}_{name}"] = result[name]
            for name in ("actions", "moves", "freeze_presses", "move_board_returns",
                         "freeze_state_returns", "forced_end_at_cap"):
                values[f"filter/{mode}_{label}_{name}"] = result["totals"].get(name, 0)
    tracker.log(values, 3200)
    metadata = json.loads((folder / "experiment.json").read_text())
    metadata["wandb_url"] = tracker.run.url
    (folder / "experiment.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(tracker.run.url, flush=True)
finally:
    tracker.finish()
