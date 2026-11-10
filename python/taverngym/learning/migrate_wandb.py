"""Backfill and tail an existing run into W&B without touching its training process."""

import argparse
import json
from pathlib import Path
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .tracking import WandbTracker, metric_name


def pending_events(events, cursors):
    grouped, updated = {}, dict(cursors)
    for name in events.Tags().get("scalars", []):
        values = events.Scalars(name)
        begin = cursors.get(name, 0)
        for event in values[begin:]:
            grouped.setdefault(event.step, {})[name] = event.value
        updated[name] = len(values)
    return sorted(grouped.items()), updated


def journal_keys(path):
    """Resume without duplicating already submitted metrics, including aliases."""
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            step = row["optimizer_step"]
            seen.update((name, step) for name in row if name != "optimizer_step")
    return seen


class EventFiles:
    """Each writer keeps its own file open; directory readers can miss updates."""

    def __init__(self, folder):
        self.folder = folder
        self.readers = {}
        self.cursors = {}

    def reload(self, seen):
        grouped, counts = {}, {}
        for path in sorted(self.folder.glob("events.out.tfevents.*")):
            name = path.name
            if name not in self.readers:
                self.readers[name] = EventAccumulator(str(path), size_guidance={"scalars": 0},
                                                     purge_orphaned_data=False)
            events = self.readers[name].Reload()
            batches, self.cursors[name] = pending_events(events, self.cursors.get(name, {}))
            for step, values in batches:
                for key, value in values.items():
                    identity = (metric_name(key), step)
                    if identity not in seen:
                        grouped.setdefault(step, {})[key] = value
                        seen.add(identity)
            for key, count in self.cursors[name].items():
                counts[key] = counts.get(key, 0) + count
        return sorted(grouped.items()), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=3)
    args = parser.parse_args()
    folder = args.experiment_dir.resolve()
    config = json.loads((folder / "config.json").read_text())
    progress = json.loads((folder / "progress.json").read_text())
    tracker = WandbTracker(folder, config, kind="live-migration", stats_pid=progress.get("pid"))
    runtime = folder / "wandb"
    state_path = runtime / "bridge-state.json"
    seen = journal_keys(folder / "wandb-metrics.jsonl")
    events = EventFiles(folder)
    quiet = 0
    try:
        while True:
            batches, cursors = events.reload(seen)
            for step, values in batches:
                tracker.log(values, step)
            status = json.loads((folder / "experiment.json").read_text()).get("status")
            state = {"status": "syncing", "source": "existing event logs; training not restarted",
                     "cursors": cursors, "train_points": cursors.get("train/loss", 0),
                     "validation_points_including_baseline": cursors.get("val/mean_placement", 0) + cursors.get("baseline/mean_placement", 0),
                     "last_sync_unix": time.time(), "run_url": tracker.run.url}
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            quiet = quiet + 1 if not batches else 0
            if status in ("completed", "failed", "interrupted", "postprocessing_failed") and quiet >= 3:
                state["status"] = "completed" if status == "completed" else status
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                tracker.run.summary.update({"actual_adam_updates": cursors.get("train/loss", 0),
                                            "val_points_including_step_zero": state["validation_points_including_baseline"]})
                break
            time.sleep(args.poll_seconds)
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish()


if __name__ == "__main__":
    main()
