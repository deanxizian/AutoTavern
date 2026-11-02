"""Opt-in V8+ output component; the active V7 continues using tracking.py.

This class does not install itself or patch the original tracker. A future run
must explicitly select it and apply the same schema to dedup/finalization keys.
"""

import numbers
import re

from .metric_schema_v2 import map_metric_payload_v2
from .tracking import WandbTracker


class WandbTrackerV2(WandbTracker):
    def __init__(self, folder, config, **kwargs):
        version = re.fullmatch(r"v(\d+)", str(config.get("model_version", "")), re.IGNORECASE)
        if version is None or int(version.group(1)) < 8:
            raise ValueError("W&B schema v2 is reserved for explicitly configured V8+ runs")
        if config.get("metric_schema_version", 2) != 2:
            raise ValueError("WandbTrackerV2 requires metric_schema_version=2")
        prepared = {**config, "metric_schema_version": 2,
                    "metric_schema_path": "configs/wandb-metric-schema-v2.json"}
        super().__init__(folder, prepared, **kwargs)

    def log(self, values, optimizer_step):
        # Match the existing writer's numeric filter; preserve scalar values and
        # its single online/local write rather than creating alias duplicates.
        numeric = {key: value for key, value in values.items() if isinstance(value, numbers.Number)}
        if "optimizer_step" in numeric and numeric["optimizer_step"] != optimizer_step:
            raise ValueError("Payload optimizer_step differs from the actual Adam axis")
        super().log(map_metric_payload_v2(numeric), optimizer_step)
