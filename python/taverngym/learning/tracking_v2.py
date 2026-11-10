"""Opt-in schema v2 for V8+ and the explicitly named 37.0.2 fresh lineage.

This class does not install itself or patch the original tracker. A future run
must explicitly select it and apply the same schema to dedup/finalization keys.
"""

import hashlib
import json
import numbers
from pathlib import Path
import re

from .metric_schema_v2 import SCHEMA_PATH, map_metric_payload_v2, metric_name_v2
from .tracking import WandbTracker


class WandbTrackerV2(WandbTracker):
    def __init__(self, folder, config, **kwargs):
        version = re.fullmatch(r"v(\d+)", str(config.get("model_version", "")), re.IGNORECASE)
        fresh = (config.get("lineage") == "eric-3702-fresh-v1"
                 and config.get("model_version") == config["lineage"])
        continuous = (config.get("lineage") == "eric-v7-3702-continuous"
                      and re.fullmatch(r"eric-v7-3702-c[0-9]{3}",
                                       str(config.get("model_version", ""))) is not None)
        if not (fresh or continuous) and (version is None or int(version.group(1)) < 8):
            raise ValueError("W&B schema v2 requires V8+ or an explicitly configured authorized 37.0.2 lineage")
        if config.get("metric_schema_version") != 2:
            raise ValueError("WandbTrackerV2 requires metric_schema_version=2")
        actual_sha = hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()
        if config.get("metric_schema_sha256") != actual_sha:
            raise ValueError("W&B schema SHA differs from explicitly sealed config")
        self.seen = {}
        journal = Path(folder) / "wandb-metrics.jsonl"
        if journal.exists():
            for line in journal.read_text().splitlines():
                row = json.loads(line)
                for key, value in row.items():
                    if key != "optimizer_step":
                        canonical = metric_name_v2(key)
                        self.seen[self.event_key(canonical, row, int(row["optimizer_step"]))] = value
        prepared = {**config, "metric_schema_version": 2,
                    "metric_schema_path": "configs/wandb-metric-schema-v2.json"}
        super().__init__(folder, prepared, **kwargs)

    def log(self, values, optimizer_step):
        # Match the existing writer's numeric filter; preserve scalar values and
        # its single online/local write rather than creating alias duplicates.
        numeric = {key: value for key, value in values.items() if isinstance(value, numbers.Number)}
        if "optimizer_step" in numeric and numeric["optimizer_step"] != optimizer_step:
            raise ValueError("Payload optimizer_step differs from the actual Adam axis")
        mapped = map_metric_payload_v2(numeric)
        fresh = {}
        for key, value in mapped.items():
            if key == "optimizer_step":
                continue
            pair = self.event_key(key, mapped, int(optimizer_step))
            scalar = value.item() if hasattr(value, "item") else value
            if pair in self.seen:
                if self.seen[pair] != scalar:
                    raise ValueError(f"Conflicting journal metric at Adam {optimizer_step}: {key}")
            else:
                fresh[key] = scalar
        if fresh:
            # Retain the PPO context in each emitted row so restart replay can
            # distinguish two legitimate PPOs sharing an Adam axis after KL stop.
            # These are existing progress fields, never an extra W&B namespace.
            for context in ("train/progress_ppo_iteration", "val/progress_ppo_iteration"):
                if context in mapped:
                    fresh[context] = mapped[context]
            super().log(fresh, optimizer_step)
            self.seen.update({self.event_key(key, mapped, int(optimizer_step)): value
                              for key, value in fresh.items()})

    @staticmethod
    def event_key(key, values, optimizer_step):
        if key.startswith("train/optimization_adam_"):
            context = None  # Each actual optimizer update remains unique.
        elif key.startswith(("train/", "train_core/", "train_reward/")):
            context = values.get("train/progress_ppo_iteration")
        elif key.startswith(("val/", "val_core/", "val_reward/")):
            context = values.get("val/progress_ppo_iteration", 0)
        else:
            context = None
        return key, optimizer_step, context


def tracker_for_config(folder, config, **kwargs):
    """Keep legacy runs unchanged; require an explicitly sealed v2 when selected."""
    version = re.fullmatch(r"v(\d+)", str(config.get("model_version", "")), re.IGNORECASE)
    modern = version is not None and int(version.group(1)) >= 8
    if modern or config.get("metric_schema_version") == 2:
        return WandbTrackerV2(folder, config, **kwargs)
    return WandbTracker(folder, config, **kwargs)


def metric_for_config(name, config):
    from .tracking import metric_name

    return (metric_name_v2(name) if config.get("metric_schema_version") == 2
            else metric_name(name))


def journal_has_metric(rows, name, config, optimizer_step=None):
    """The writer, repair and finalizer share one canonical marker mapping."""
    marker = metric_for_config(name, config)
    return any(marker in row and (optimizer_step is None
               or row.get("optimizer_step") == optimizer_step) for row in rows)
