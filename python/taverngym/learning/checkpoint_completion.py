"""Persisted optimization-completion evidence for resumable PPO checkpoints."""

import hashlib
import json
from pathlib import Path
import zipfile


COMPLETION_ATTRIBUTE = "_ppo_completion"


def ppo_completion_marker(model, status):
    """Capture actual clocks, including variable Adam counts after early KL stopping."""
    return {
        "version": 1,
        "status": status,
        "num_timesteps": int(model.num_timesteps),
        "n_updates": int(model._n_updates),
        "source_Adam_offset": int(getattr(model, "_optimizer_step_offset", 0)),
        "stage_Adam": model.optimizer_step_count(),
        "total_Adam": model.optimizer_total_step_count(),
    }


def ensure_complete_ppo_checkpoint(model, path):
    """Reject interrupted rollouts/updates before resume or stage counter reset.

    Newly saved EntityPPO archives carry completion evidence. Positive-step legacy
    archives require a SHA-bound ``<checkpoint>.boundary.json`` attestation; a
    sidecar can never override an explicit in-progress or invalid marker.
    """
    path = Path(path)
    if not path.is_file() and path.suffix != ".zip":
        path = Path(str(path) + ".zip")
    with zipfile.ZipFile(path) as archive:
        data = json.loads(archive.read("data"))
    actual = ppo_completion_marker(model, "complete")
    current = getattr(model, COMPLETION_ATTRIBUTE, None)
    if isinstance(current, dict) and current.get("status") == "in_progress":
        raise ValueError("Checkpoint PPO optimization/collection is in progress")
    for key, saved_key in (("num_timesteps", "num_timesteps"), ("n_updates", "_n_updates"),
                           ("source_Adam_offset", "_optimizer_step_offset")):
        saved = data.get(saved_key, 0)
        if type(saved) is not int or saved != actual[key]:
            raise ValueError(f"Checkpoint/model clock differs: {key}")
    if (min(actual[key] for key in ("num_timesteps", "n_updates", "source_Adam_offset",
                                   "stage_Adam", "total_Adam")) < 0
            or actual["total_Adam"] != actual["source_Adam_offset"] + actual["stage_Adam"]):
        raise ValueError("Checkpoint Adam/source clocks are inconsistent")
    marker = data.get(COMPLETION_ATTRIBUTE)
    if marker is not None:
        if (marker != actual or current != marker
                or any(type(marker[k]) is not type(v) for k, v in actual.items())):
            raise ValueError("Checkpoint lacks a matching complete PPO optimization marker")
        return {"complete": True, "evidence": "persisted_completion_marker", **actual}
    # SB3's loading constructor creates a default marker even for an old ZIP.
    # Therefore inspect the archive itself, never treat that default as evidence.
    if actual["num_timesteps"] == actual["n_updates"] == actual["stage_Adam"] == 0:
        return {"complete": True, "evidence": "zero_step_initialization", **actual}
    config = data.get("experiment_config", {})
    decisions = config.get("decisions_per_iteration")
    if decisions is None:
        decisions = data.get("n_envs", 0) * data.get("n_steps", 0)
    if (type(decisions) is not int or decisions <= 0
            or actual["num_timesteps"] % decisions):
        raise ValueError("Legacy checkpoint is not a complete PPO rollout boundary")
    sidecar = Path(str(path) + ".boundary.json")
    if not sidecar.is_file():
        raise ValueError("Legacy positive-step checkpoint requires SHA-bound complete PPO evidence")
    boundary = json.loads(sidecar.read_text())
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    expected = {"samples": actual["num_timesteps"],
                "source_Adam_offset": actual["source_Adam_offset"],
                "stage_Adam": actual["stage_Adam"]}
    if (boundary.get("complete_PPO_boundary") is not True
            or boundary.get("train_returned") is not True
            or boundary.get("checkpoint_sha256") != digest
            or any(type(boundary.get(k)) is not int or boundary[k] != v
                   for k, v in expected.items())
            or ("PPO" in boundary and boundary["PPO"] != actual["num_timesteps"] // decisions)):
        raise ValueError("Legacy checkpoint boundary SHA/samples/Adam/completion evidence differs")
    return {"complete": True, "evidence": "SHA_bound_legacy_boundary", **actual}
