"""Native W&B metrics with flat names and an auditable local JSONL journal."""

import json
import numbers
import os
from pathlib import Path

from stable_baselines3.common.logger import KVWriter


def metric_name(name):
    aliases = {"baseline/": "val/", "rollout/": "train/", "eval/": "val/",
               "ppo_summary/": "train/ppo_", "time/": "system/",
               "progress/ppo_iterations": "train/ppo_iteration",
               "progress/sampled_decisions": "train/sampled_decisions"}
    if name in aliases:
        return aliases[name]
    for prefix, replacement in aliases.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):].replace("/", "_")
    head, separator, tail = name.partition("/")
    return head + separator + tail.replace("/", "_")


def tracking_settings(config):
    path = Path(__file__).resolve().parents[3] / "configs/tracking.json"
    defaults = json.loads(path.read_text()) if path.exists() else {"backend": "wandb", "project": "AutoTavern", "mode": "online"}
    return {**defaults, **config.get("tracking", {})}


class WandbTracker:
    def __init__(self, folder, config, *, kind="native", stats_pid=None, mode=None):
        import wandb

        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.kind = kind
        settings = tracking_settings(config)
        mode = mode or settings.get("mode", "online")
        metadata_path = self.folder / "wandb.json"
        previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        options = {"console": "off", "disable_git": True, "disable_code": True,
                   "init_timeout": 60, "x_stats_pid": stats_pid or os.getpid()}
        self.run = wandb.init(entity=settings.get("entity"), project=settings.get("project", "AutoTavern"),
                              name=config.get("name", self.folder.name), id=previous.get("run_id"),
                              resume="allow" if previous.get("run_id") else None,
                              config=config, dir=str(self.folder), mode=mode, save_code=False,
                              settings=wandb.Settings(**options))
        self.run.define_metric("optimizer_step")
        self.run.define_metric("*", step_metric="optimizer_step")
        metadata = {"backend": "wandb", "kind": kind, "mode": mode,
                    "entity": self.run.entity, "project": self.run.project,
                    "run_id": self.run.id, "url": self.run.url,
                    "x_axis": "optimizer_step", "weave_project": settings.get("weave_project"),
                    "weave_initialized": False}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if mode == "online" and settings.get("weave_project"):
            import weave
            self.weave_client = weave.init(settings["weave_project"])
            metadata["weave_initialized"] = True
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        self.journal = (self.folder / "wandb-metrics.jsonl").open("a", buffering=1)
        self.closed = False

    def log(self, values, optimizer_step):
        payload = {metric_name(name): value.item() if hasattr(value, "item") else value
                   for name, value in values.items() if isinstance(value, numbers.Number)}
        if not payload:
            return
        payload["optimizer_step"] = int(optimizer_step)
        # W&B's internal history row always increases; the explicit x axis also
        # permits a late validation result at an earlier optimizer step.
        self.run.log(payload)
        self.journal.write(json.dumps(payload, allow_nan=True) + "\n")

    def finish(self, exit_code=0):
        if not self.closed:
            self.journal.close()
            self.run.finish(exit_code=exit_code)
            self.closed = True


class WandbOutputFormat(KVWriter):
    def __init__(self, tracker):
        self.tracker = tracker

    def write(self, key_values, key_excluded, step=0):
        actual_step = int(key_values.get("train/optimizer_steps", step))
        values = {name: value for name, value in key_values.items()
                  if "wandb" not in key_excluded.get(name, ())}
        self.tracker.log(values, actual_step)

    def close(self):
        pass
