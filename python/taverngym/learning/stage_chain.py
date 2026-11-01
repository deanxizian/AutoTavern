"""Supervise a locked, finite sequence of PPO stages and independent finalization.

The manifest fixes the authorized order and budgets. Recovery is a separate,
explicitly diagnosed invocation; signals never cause an automatic restart.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .experiment import now, write_json
from .rewards import reward_protocols_equal


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value):
    serialized = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(serialized.encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def instance_lock(folder):
    """Kernel lock remains authoritative even when a stale PID file survives."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "controller.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another controller holds this instance lock") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps({"pid": os.getpid(), "acquired_at": now()}))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def tensor_digest(value):
    import torch

    result = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            result.update(f"{item.dtype}:{tuple(item.shape)}:".encode())
            result.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(repr(key).encode())
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for part in item:
                add(part)
        else:
            result.update(repr(item).encode())

    add(value)
    return result.hexdigest()


def inspect_checkpoint(path):
    """Inspect the saved policy/Adam directly, without allocating a training env."""
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file

    data, params, _ = load_from_zip_file(path, device="cpu")
    require("policy" in params and "policy.optimizer" in params, "Incomplete PPO archive")
    optimizer = params["policy.optimizer"]
    require(bool(optimizer["state"]), "Missing trained Adam state")
    counters = []
    for state in optimizer["state"].values():
        require("exp_avg" in state and "exp_avg_sq" in state, "Incomplete Adam moments")
        counters.append(int(state["step"]))
        require(all(torch.isfinite(v).all().item() for v in state.values()
                    if torch.is_tensor(v)), "Nonfinite Adam checkpoint")
    require(len(set(counters)) == 1, "Inconsistent Adam clocks")
    require(all(torch.isfinite(v).all().item() for v in params["policy"].values()
                if torch.is_tensor(v)), "Nonfinite policy checkpoint")
    budget = data.get("experiment_config", {}).get("decisions_per_iteration")
    require(budget and data["num_timesteps"] % budget == 0, "Checkpoint is not a PPO boundary")
    require(data.get("_n_updates") == data["num_timesteps"] // budget * data["n_epochs"],
            "Checkpoint contains incomplete PPO optimizer epochs")
    total = counters[0]
    return {
        "path": str(Path(path).resolve()), "sha256": digest(path),
        "bytes": Path(path).stat().st_size,
        "ppo_iteration": data["num_timesteps"] // budget,
        "sampled_decisions": data["num_timesteps"],
        "stage_optimizer_steps": total - data.get("_optimizer_step_offset", 0),
        "adam_total_steps": total, "adam_states": len(counters),
        "learning_rates": [g["lr"] for g in optimizer["param_groups"]],
        "policy_sha256": tensor_digest(params["policy"]),
        "adam_moments_sha256": tensor_digest(optimizer["state"]),
        "model_version": data.get("experiment_config", {}).get("model_version"),
        "reward_shaping": data.get("experiment_config", {}).get("reward_shaping"),
    }


def validation_selection(folder, config):
    expected = [0, *range(config["evaluation_every_iterations"], config["iterations"] + 1,
                          config["evaluation_every_iterations"])]
    rows = []
    require({int(p.stem.rsplit("-", 1)[1]) for p in
             folder.glob("evaluation-[0-9][0-9][0-9][0-9].json")} == set(expected),
            "Unexpected/missing validation iteration set")
    for iteration in expected:
        result = read_json(folder / f"evaluation-{iteration:04d}.json")
        validate_evaluation(result, config["evaluation_games"], config["evaluation_seed"],
                            config["opponent_pool"])
        rows.append((iteration, result))
    # min retains the earliest row on ties. Heldout files are never considered.
    return min(rows, key=lambda row: row[1]["mean_placement"]), rows


def validate_evaluation(result, games, seed, opponent_pool=None):
    require(math.isfinite(result["mean_placement"]), "Nonfinite evaluation")
    require(set(result["conditions"]) == {"mixed_sampled", "mixed_greedy"},
            "Unexpected evaluation conditions")
    require(result.get("selection_condition") == "mixed_sampled"
            and result["mean_placement"] == result["conditions"]["mixed_sampled"]["mean_placement"],
            "Selection condition differs from mixed sampled validation")
    for name, row in result["conditions"].items():
        require(row["games"] == games and len(row["placements"]) == games,
                "Incomplete evaluation game count")
        require(row["seed"] == seed and row["deterministic"] == (name == "mixed_greedy"),
                "Evaluation protocol mismatch")
        require(all(1 <= placement <= 8 for placement in row["placements"]),
                "Invalid real terminal placements")
        require(row["mean_placement"] == sum(row["placements"]) / games,
                "Reported placement disagrees with actual games")
        expected = ([(g["name"], g["count"], False) for g in opponent_pool]
                    if opponent_pool is not None else [("bot_v1", 2, False), ("bot_v2", 3, False)])
        require([(g["name"], g["count"], g["deterministic"])
                 for g in row["opponent_groups"]] == expected,
                "Evaluation opponent roster changed")
        if opponent_pool is not None:
            require(all(reward_protocols_equal(group.get("reward_shaping"), spec.get("reward_shaping"))
                        for group, spec in zip(row["opponent_groups"], opponent_pool)),
                    "Evaluation opponent termination protocol changed")


def verify_config(config):
    fixed = {"iterations": 100, "decisions_per_iteration": 131072, "n_envs": 128,
             "n_steps": 1024, "batch_size": 8192, "micro_batch_size": 1024, "n_epochs": 1,
             "learning_rate_start": 1e-5, "learning_rate_end": 1e-5,
             "initial_evaluation_games": 64, "evaluation_games": 64,
             "evaluation_every_iterations": 50, "final_evaluation_games": 128,
             "evaluation_include_scripted": False, "preserve_optimizer_state": True,
             "evaluation_seed": 40000003, "final_evaluation_seed": 50000003,
             "learning_rate_schedule": "linear", "forward_precision": "bf16",
             "opponent_forward_precision": "bf16",
             "sampling_driver": "unified_actor", "checkpoint_every_iterations": 50}
    require(all(config.get(key) == value for key, value in fixed.items()),
            "Configuration differs from this instance's authorized protocol")
    expected_reward = config.get("expected_reward_shaping", {
        "operation_cost": 0, "cycle_cost": .0002, "cycle_detection": "all_actions",
        "cycle_forfeit_repetitions": 3, "turn_action_forfeit_limit": 60})
    require(reward_protocols_equal(config["reward_shaping"], expected_reward),
            "Reward/termination protocol changed")
    expected = config.get("expected_opponent_counts", {"bot_v1": 2, "bot_v2": 3})
    require({s["name"]: s["count"] for s in config["opponent_pool"]} == expected,
            "Opponent roster changed")
    require(config["frozen_opponents"] == sum(expected.values()), "Opponent seat total changed")


def bind_source(root, folder, previous, known_results=None):
    """Bind only a completed predecessor's selected complete PPO/Adam archive."""
    config = read_json(folder / "config.json")
    selected = previous["retained_model"]
    source = Path(selected["path"])
    source = source if source.is_absolute() else root / source
    actual = inspect_checkpoint(source)
    require(actual == selected, "Previous-stage selected checkpoint changed")
    require(previous["status"] == "completed", "Predecessor finalization incomplete")
    if config.get("initial_model"):
        require((root / config["initial_model"]).resolve() == source.resolve()
                and config["initial_model_sha256"] == actual["sha256"],
                "Refusing to overwrite a different bound source")
    history = folder / "preparation-history" / "before-source-binding"
    history.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "experiment.json"):
        if not (history / name).exists():
            shutil.copy2(folder / name, history / name)
    config.update(initial_model=str(source.relative_to(root)),
                  initial_model_sha256=actual["sha256"],
                  initial_model_version=previous["model_version"],
                  initial_model_ppo_iteration=actual["ppo_iteration"],
                  source_adam_total_steps=actual["adam_total_steps"], config_status="source_bound")
    config["derived_budget"].update(source_adam_updates=actual["adam_total_steps"],
        final_total_adam_updates_max=actual["adam_total_steps"] + 1600)
    known = {previous["model_version"]: previous, **(known_results or {})}
    for spec in config["opponent_pool"]:
        if not spec.get("source_stage"):
            continue
        result = known[spec["source_stage"]]
        selected_opponent = result["retained_model"]
        require(inspect_checkpoint(selected_opponent["path"]) == selected_opponent,
                "Selected frozen opponent checkpoint changed")
        relative = str(Path(selected_opponent["path"]).relative_to(root))
        require(not spec.get("model") or (spec["model"] == relative
                and spec["sha256"] == selected_opponent["sha256"]),
                "Refusing to change an already bound opponent")
        spec.update(model=relative, sha256=selected_opponent["sha256"],
                    reward_shaping=result["reward_shaping"],
                    protocol_source=result["config_path"],
                    model_version=result["model_version"])
    binding = {"bound_at": now(), "source": actual,
        "previous_stage": previous["model_version"], "selection": previous["selection"],
        "config_sha256": json_digest(config), "heldout_used_for_selection": False,
        "previous_config_sha256": digest(folder / "config.json"),
        "status": "pending"}
    # The intent is durable before changing the config, so either side of that
    # atomic rename can be verified and completed after a controller crash.
    write_json(folder / "source-binding.json", binding)
    write_json(folder / "config.json", config)
    metadata = read_json(folder / "experiment.json")
    metadata.update(status="source_bound", config_sha256=digest(folder / "config.json"),
                    source_model=actual, source_binding="source-binding.json")
    write_json(folder / "experiment.json", metadata)
    binding["status"] = "completed"
    write_json(folder / "source-binding.json", binding)
    return config


def no_live_stage_process(folder):
    """Also find a child orphaned between Popen and persisting its PID."""
    modules = {"taverngym.learning.experiment", "taverngym.learning.stage_chain"}
    targets = {str(folder), str(folder / "config.json")}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        try:
            args = entry.joinpath("cmdline").read_bytes().decode().split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        require(not (modules.intersection(args) and targets.intersection(args)),
                f"Stage process {entry.name} is still alive; refusing duplicate execution")


def validate_sealed_lifecycle_ranking(root, manifest, previous, known_results, inputs):
    """Recover from the sealed lifecycle snapshot without opening its global file."""
    from .stage_extension import validate_lifecycle_input

    require(inputs.get("model_version_registry") == manifest["model_version_registry"],
            "Sealed ranking registry path changed")
    validate_lifecycle_input(inputs)
    registry_hash = inputs.get("model_version_registry_sha256", "")
    require(len(registry_hash) == 64 and set(registry_hash) <= set("0123456789abcdef"),
            "Sealed ranking registry hash missing or invalid")
    require(inputs.get("pool_selection") == manifest.get("pool_selection") == "top3"
            and inputs.get("protected_version") is None
            and inputs.get("learning_source_version") == previous["model_version"]
            and inputs["protocol"] == manifest["ranking_protocol"],
            "Sealed ranking source/protocol changed during recovery")
    require(set(manifest.get("excluded_opponent_versions", []))
            <= set(inputs["excluded_opponent_versions"]), "Sealed lifecycle exclusions changed")
    source = previous["retained_model"]
    require(digest(root / source["path"]) == source["sha256"], "Ranking learning source changed")
    registry = {version: dict(spec) for version, spec in manifest["historical_models"].items()}
    for version, result in {previous["model_version"]: previous, **known_results}.items():
        model = result["retained_model"]
        registry[version] = {"name": f"bot_{version}", "count": 2,
            "model": str(Path(model["path"]).relative_to(root)), "sha256": model["sha256"],
            "model_version": version, "source_stage": version,
            "reward_shaping": result["reward_shaping"], "protocol_source": result["config_path"]}
    order = inputs["protocol"]["version_order"]
    prefix = order[:order.index(previous["model_version"]) + 1]
    lifecycle = inputs["model_version_registry_snapshot"]["versions"]
    require(all(version in lifecycle for version in prefix), "Sealed lifecycle history incomplete")
    active = [version for version in prefix if lifecycle[version]["status"] == "active"]
    require(all(version in registry and registry[version]["model_version"] == version
                for version in active), "Active ranking source missing or aliased")
    require(inputs["candidates"] == [registry[version] for version in active],
            "Sealed ranking candidates changed during recovery")
    for spec in [*inputs["candidates"], *inputs["protocol"]["reference_pool"]]:
        require(digest(root / spec["model"]) == spec["sha256"],
                "Sealed ranking candidate/reference checkpoint changed")


def complete_retention(folder):
    """Finish an already verified cleanup manifest, including after a crash."""
    manifest = read_json(folder / "retention.json")
    selected = manifest["kept_model"]
    require(digest(selected["path"]) == selected["sha256"], "Retained best changed")
    if manifest["status"] != "completed":
        root = next(p for p in folder.parents if (p / "pyproject.toml").is_file())
        config = read_json(folder / "resolved-config.json")
        models = root / "runs" / config["name"]
        for item in manifest["removed_models"]:
            path = root / item["path"]
            require(not path.is_symlink() and path.parent in (models, models / "checkpoints"),
                    "Unsafe retention manifest path")
            if path.exists():
                require(digest(path) == item["sha256"], "Cleanup candidate changed")
                path.unlink()
        manifest.update(status="completed", completed_at=now())
        write_json(folder / "retention.json", manifest)
    metadata = read_json(folder / "experiment.json")
    metadata.update(status="completed", completed_at=now(), retained_model=selected,
                    results="results.json")
    write_json(folder / "experiment.json", metadata)
    progress = read_json(folder / "progress.json")
    progress.update(status="completed", phase="completed", updated_at=now(),
                    latest_model=selected["path"])
    write_json(folder / "progress.json", progress)


def finalize(root, folder, device):
    import torch
    from .experiment import evaluate_suite, load_frozen_opponents
    from .ppo import EntityPPO, PlacementBuffer
    from .rewards import REWARD_METRICS
    from .tracking import WandbTracker

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    config = read_json(folder / "resolved-config.json")
    verify_config(config)
    progress = read_json(folder / "progress.json")
    require(progress["status"] == "completed" and progress["completed_iterations"] == 100,
            "Training is incomplete")
    models = root / "runs" / config["name"]
    source = root / config["initial_model"]
    require(digest(source) == config["initial_model_sha256"], "Initial source hash changed")
    for spec in config["opponent_pool"]:
        require(digest(root / spec["model"]) == spec["sha256"], "Opponent hash changed")
    (iteration, evaluation), evaluations = validation_selection(folder, config)
    best = models / config["best_model_filename"]
    selected = inspect_checkpoint(best)
    require(selected["ppo_iteration"] == iteration,
            "Best disagrees with strict validation selection")
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    require([r["iteration"] for r in rows] == list(range(1, 101)), "Missing/duplicate PPO rows")
    for row in rows:
        require(row["decisions"] == row["iteration"] * 131072, "Sample budget mismatch")
        require(row["train/learning_rate"] == 1e-5, "LR changed during stage")
    steps = int(rows[-1]["train/optimizer_steps"])
    require(steps == sum(int(r["train/optimizer_steps_this_iteration"]) for r in rows)
            and steps <= 1600, "Adam budget mismatch")
    journal_rows = [json.loads(line) for line in
                    (folder / "wandb-metrics.jsonl").read_text().splitlines()]
    finite_metrics = all(math.isfinite(value) for row in [*rows, *journal_rows]
                         for value in row.values() if isinstance(value, (float, int)))
    require(finite_metrics, "Nonfinite training or tracking metric")
    require(selected["stage_optimizer_steps"] == (0 if iteration == 0 else
            int(rows[iteration - 1]["train/optimizer_steps"])), "Selected stage Adam mismatch")
    require(selected["adam_total_steps"] == config["stage_source_adam_total_steps"]
            + selected["stage_optimizer_steps"], "Selected inherited Adam mismatch")
    starting = inspect_checkpoint(models / "starting_model.zip")
    original = inspect_checkpoint(source)
    require(starting["policy_sha256"] == original["policy_sha256"]
            and starting["adam_moments_sha256"] == original["adam_moments_sha256"],
            "Stage initialization lost source weights/Adam")
    require(starting["ppo_iteration"] == 0 and starting["stage_optimizer_steps"] == 0
            and starting["adam_total_steps"] == original["adam_total_steps"]
            and starting["learning_rates"] == [1e-5], "Stage-zero counters/LR mismatch")
    final_info = inspect_checkpoint(models / "final_model.zip")
    require(final_info["ppo_iteration"] == 100 and final_info["stage_optimizer_steps"] == steps
            and final_info["adam_total_steps"] == original["adam_total_steps"] + steps,
            "Final checkpoint budget/Adam mismatch")
    boundary = inspect_checkpoint(models / ("starting_model.zip" if iteration == 0
                    else f"checkpoints/iteration-{iteration:04d}.zip"))
    require(selected["policy_sha256"] == boundary["policy_sha256"]
            and selected["adam_moments_sha256"] == boundary["adam_moments_sha256"],
            "Best does not match the selected full PPO/Adam boundary")
    require(selected["learning_rates"] == [1e-5], "Best LR does not match fixed schedule")
    opponents = load_frozen_opponents(config, device)
    heldout = {}
    fingerprints = {}
    paths = {"final": models / "final_model.zip", "initial": models / "starting_model.zip",
             "best": best}
    for name, path in paths.items():
        info = inspect_checkpoint(path)
        output = folder / f"evaluation-{name}-heldout.json"
        if output.exists():
            result = read_json(output)
            require(result.get("policy_sha256", info["policy_sha256"]) == info["policy_sha256"],
                    "Existing heldout was evaluated on different policy weights")
        elif info["policy_sha256"] in fingerprints:
            other = fingerprints[info["policy_sha256"]]
            result = {**heldout[other], "reused_from": other}
        else:
            model = EntityPPO.load(path, device=device, n_envs=1, n_steps=2, batch_size=2,
                                  rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
            result = evaluate_suite(model, 128, config["final_evaluation_seed"], config, opponents)
            del model
        validate_evaluation(result, 128, config["final_evaluation_seed"], config["opponent_pool"])
        result.update(model=name, stage_optimizer_steps=info["stage_optimizer_steps"],
                      policy_sha256=info["policy_sha256"], used_for_selection=False)
        write_json(output, result)
        heldout[name] = result
        fingerprints[info["policy_sha256"]] = name
    tracker = WandbTracker(folder, config)
    try:
        journal = [json.loads(line) for line in
                   (folder / "wandb-metrics.jsonl").read_text().splitlines()]
        for name, result in heldout.items():
            marker = f"test/{name}_mixed_sampled_mean_placement"
            if any(marker in row for row in journal):
                continue
            values = {f"test/{name}_{condition}_{key}": row[key]
                      for condition, row in result["conditions"].items()
                      for key in ("mean_placement", "first_rate", "top4_rate", "cap_end_rate",
                                  *REWARD_METRICS) if key in row}
            tracker.log(values, result["stage_optimizer_steps"])
    finally:
        tracker.finish()
    # Cleanup is restricted to this stage's outputs, after all independent evaluations.
    protected = {source.resolve(), best.resolve(),
                 *((root / s["model"]).resolve() for s in config["opponent_pool"])}
    candidates = [p for p in models.rglob("*.zip") if p.resolve() not in protected]
    require(all(not p.is_symlink() and p.parent in (models, models / "checkpoints")
                for p in candidates), "Unsafe retention candidate")
    removed = [{"path": str(p.relative_to(root)), "sha256": digest(p),
                "bytes": p.stat().st_size} for p in candidates]
    manifest = {"status": "planned", "kept_model": selected, "removed_models": removed,
                "heldout_used_for_selection": False}
    write_json(folder / "retention.json", manifest)
    result = {"status": "completed", "completed_at": now(),
              "model_version": config["model_version"],
              "reward_shaping": config["reward_shaping"],
              "config_path": str(folder / "config.json"), "opponent_pool": config["opponent_pool"],
              "fresh_samples": 13107200, "actual_adam_updates": steps,
              "retained_model": selected, "source_model": original, "heldout": heldout,
              "selection": {"ppo_iteration": iteration,
                            "mean_placement": evaluation["mean_placement"],
                            "criterion": "mixed sampled strict decrease; earliest tie",
                            "validation_iterations": [i for i, _ in evaluations]},
              "initial_model_unchanged": digest(source) == config["initial_model_sha256"],
              "verification": {"finite_and_protocol_consistent": finite_metrics,
                               "full_checkpoints_and_adam_verified": True},
              "weights_uploaded": False, "heldout_used_for_selection": False}
    write_json(folder / "results.json", result)
    complete_retention(folder)


def repair_boundary(root, folder, device):
    """Settle a complete PPO checkpoint interrupted before validation/metrics commit."""
    import torch
    from .experiment import evaluate_suite, load_frozen_opponents, log_evaluation, save_model
    from .ppo import EntityPPO, PlacementBuffer
    from .tracking import WandbTracker

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    config = read_json(folder / "resolved-config.json")
    models = root / "runs" / config["name"]
    checkpoint = models / "latest_model.zip"
    if not checkpoint.exists():
        checkpoint = models / "starting_model.zip"
    info = inspect_checkpoint(checkpoint)
    iteration = info["ppo_iteration"]
    model = EntityPPO.load(checkpoint, device=device, n_envs=1, n_steps=2, batch_size=2,
                          rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
    opponents = load_frozen_opponents(config, device)
    if iteration % config["evaluation_every_iterations"] == 0:
        output = folder / f"evaluation-{iteration:04d}.json"
        if not output.exists():
            result = evaluate_suite(model, 64, config["evaluation_seed"], config, opponents)
            write_json(output, result)
        result = read_json(output)
        validate_evaluation(result, 64, config["evaluation_seed"], config["opponent_pool"])
        earlier = [read_json(folder / f"evaluation-{i:04d}.json")["mean_placement"]
                   for i in range(0, iteration, config["evaluation_every_iterations"])]
        if not earlier or result["mean_placement"] < min(earlier):
            save_model(model, models / config["best_model_filename"])
        journal = [json.loads(line) for line in
                   (folder / "wandb-metrics.jsonl").read_text().splitlines()]
        if not any("val/mean_placement" in r
                   and r["optimizer_step"] == info["stage_optimizer_steps"]
                   for r in journal):
            from stable_baselines3.common.logger import configure
            from .tracking import WandbOutputFormat

            tracker = WandbTracker(folder, config)
            try:
                logger = configure(format_strings=[])
                logger.output_formats.append(WandbOutputFormat(tracker))
                log_evaluation(logger, "val", result)
                logger.record("val/ppo_iteration", iteration)
                logger.dump(info["stage_optimizer_steps"])
            finally:
                tracker.finish()
    metrics = folder / "metrics.jsonl"
    rows = ([json.loads(line) for line in metrics.read_text().splitlines()]
            if metrics.exists() else [])
    require([r["iteration"] for r in rows] in (list(range(1, iteration)),
            list(range(1, iteration + 1))), "Recovery requires contiguous committed metrics")
    if len(rows) == iteration - 1:
        row = {"iteration": iteration, "decisions": info["sampled_decisions"],
               "train/optimizer_steps": info["stage_optimizer_steps"],
               "train/optimizer_steps_this_iteration": info["stage_optimizer_steps"]
                    - (int(rows[-1]["train/optimizer_steps"]) if rows else 0),
               "train/learning_rate": 1e-5, "rollout_seconds": model.last_rollout_seconds,
               "update_seconds": model.last_train_seconds, "recovered_boundary": True,
               "unrecoverable_metrics": "original rollout outcome/loss absent; not reconstructed",
               }
        with metrics.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    progress = read_json(folder / "progress.json")
    if checkpoint.name == "starting_model.zip":
        shutil.copy2(checkpoint, models / "latest_model.zip")
    if iteration == config["iterations"]:
        final = models / "final_model.zip"
        if not final.exists():
            shutil.copy2(checkpoint, final)
        output = folder / "evaluation-final-heldout.json"
        if not output.exists():
            result = evaluate_suite(model, 128, config["final_evaluation_seed"], config, opponents)
            write_json(output, result)
        progress.update(status="completed", phase="training_complete",
                        completed_iterations=iteration, sampled_decisions=info["sampled_decisions"],
                        optimizer_steps=info["stage_optimizer_steps"],
                        updated_at=now())
        write_json(folder / "progress.json", progress)
    write_json(folder / "boundary-repair.json", {"repaired_at": now(), "checkpoint": info,
               "completed_iterations": iteration, "fresh_samples_added": 0,
               "adam_updates_added": 0})


class StageChain:
    def __init__(self, manifest_path, resume=False, recovery_reason=None):
        self.manifest_path = Path(manifest_path).resolve()
        self.folder = self.manifest_path.parent
        self.manifest = read_json(self.manifest_path)
        self.root = Path(self.manifest["root"])
        self.resume = resume
        self.recovery_reason = recovery_reason
        self.stopping = False
        self.child = None
        self.state_path = self.folder / "controller-status.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {}
        self.previous_child_pid = self.state.get("training_pid")
        self.state.update(instance_id=self.manifest["instance_id"], controller_pid=os.getpid(),
                          manifest=str(self.manifest_path), updated_at=now())

    def status(self, **changes):
        self.state.update(changes, updated_at=now())
        write_json(self.state_path, self.state)

    def signal(self, signum, frame):
        self.stopping = True
        self.status(status="stopping",
                    stop_reason=f"explicit {signal.Signals(signum).name}; no automatic restart")
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)

    def budget(self, index, progress):
        completed = self.state.setdefault("completed_stages", [])
        used_ppo = (sum(s["ppo_iterations"] for s in completed)
                    + progress.get("completed_iterations", 0))
        used_samples = sum(s["samples"] for s in completed) + progress.get("sampled_decisions", 0)
        used_adam = sum(s["adam_updates"] for s in completed) + progress.get("optimizer_steps", 0)
        maximum = (self.manifest["budget"] if self.state.get("conditional_extension_enabled")
                   else self.manifest.get("primary_budget", self.manifest["budget"]))
        require(used_ppo <= maximum["ppo_iterations"] and used_samples <= maximum["samples"]
                and used_adam <= maximum["adam_updates"], "Authorized total budget exceeded")
        return {"used": {"ppo_iterations": used_ppo, "samples": used_samples,
                         "adam_updates": used_adam},
                "remaining": {"ppo_iterations": maximum["ppo_iterations"] - used_ppo,
                              "samples": maximum["samples"] - used_samples,
                              "adam_updates": maximum["adam_updates"] - used_adam}}

    def run_child(self, command, stage, index, phase):
        require(not self.stopping, "Explicit stop requested")
        folder = self.root / stage["experiment_dir"]
        log = folder / f"{phase}-{time.time_ns()}.log"
        with log.open("x", buffering=1) as stream:
            self.child = subprocess.Popen(command, cwd=self.root, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            self.status(status="running", phase=phase, stage=stage["version"], stage_index=index,
                        training_pid=self.child.pid, log_path=str(log), command=command,
                        config_path=str(folder / "config.json"), checkpoint_directory=str(
                            self.root / "runs" / read_json(folder / "config.json")["name"]))
            try:
                while self.child.poll() is None:
                    progress_path = folder / "progress.json"
                    progress = read_json(progress_path) if progress_path.exists() else {}
                    self.status(progress=progress, budget=self.budget(index, progress),
                                complete_checkpoint=progress.get("latest_model"),
                                last_progress_at=progress.get("updated_at"))
                    time.sleep(5)
            except BaseException:
                if self.child.poll() is None:
                    os.killpg(self.child.pid, signal.SIGTERM)
                    self.child.wait()
                raise
            code = self.child.returncode
            self.status(last_child_exit_code=code, training_pid=None)
            self.child = None
        require(not self.stopping, "Controller stopped by explicit signal; no automatic restart")
        require(code == 0, f"{phase} failed with exit {code}; inspect {log}")

    def command(self, mode, folder):
        return [sys.executable, "-u", "-m", "taverngym.learning.stage_chain",
                "--manifest", str(self.manifest_path),
                "--worker", mode, "--experiment-dir", str(folder)]

    def recover(self, stage, index, folder, config):
        require(self.resume and self.recovery_reason,
                "Incomplete stage requires diagnosed --resume --recovery-reason")
        count = self.state.setdefault("recoveries", {}).get(stage["version"], 0)
        require(count < 1, "Stage automatic recovery limit of one exhausted")
        require(not (folder / "last-interruption-signal.json").exists(),
                "Stage received a stop signal; automatic recovery forbidden")
        checkpoint = self.root / "runs" / config["name"] / "latest_model.zip"
        if not checkpoint.exists():
            checkpoint = checkpoint.with_name("starting_model.zip")
        info = inspect_checkpoint(checkpoint)
        progress = read_json(folder / "progress.json")
        require(progress.get("sampled_decisions", 0) <= info["sampled_decisions"]
                and progress.get("optimizer_steps", 0) <= info["stage_optimizer_steps"],
                "Discarded partial work would exceed the authorized fixed sample/Adam budget")
        require(progress.get("phase") in ("evaluating", "initial_evaluation", "iteration_complete"),
                "Cannot prove zero discarded partial work from this interrupted phase")
        history = folder / "recovery-history" / str(time.time_ns())
        history.mkdir(parents=True)
        for name in ("progress.json", "training.json", "metrics.jsonl", "wandb.json",
                     "resolved-config.json"):
            if (folder / name).exists():
                shutil.copy2(folder / name, history / name)
        write_json(history / "diagnosis.json", {"reason": self.recovery_reason,
                   "at": now(), "checkpoint": info})
        self.state["recoveries"][stage["version"]] = count + 1
        self.status(recovery_reason=self.recovery_reason)
        self.run_child(self.command("repair", folder), stage, index, "boundary-repair")
        if info["ppo_iteration"] < config["iterations"]:
            self.train(stage, index, folder, config, resume=True)

    def train(self, stage, index, folder, config, resume=False):
        checkpoint = (self.root / "runs" / config["name"] / "latest_model.zip" if resume
                      else self.root / config["initial_model"])
        command = [sys.executable, "-u", "-m", "taverngym.learning.experiment", "--config",
                   str(folder / "config.json"), "--model", str(checkpoint),
                   "--experiment-dir", str(folder),
                   "--run-dir", str(self.root / "runs" / config["name"]), "--device", "cuda",
                   "--resume" if resume else "--continue-stage"]
        metadata = read_json(folder / "experiment.json")
        metadata.update(status="running", training_started=True, started_at=now(),
                        instance_id=self.manifest["instance_id"], supervisor_pid=os.getpid())
        write_json(folder / "experiment.json", metadata)
        self.run_child(command, stage, index, "training")
        require(read_json(folder / "progress.json")["status"] == "completed",
                "Training did not complete")

    def run(self):
        with instance_lock(self.folder):
            if self.previous_child_pid:
                command = Path(f"/proc/{self.previous_child_pid}/cmdline")
                require(not command.exists() or b"taverngym.learning" not in command.read_bytes(),
                        "Previous controller child is still alive; refusing duplicate execution")
            previous_status = self.state.get("status")
            require(previous_status not in ("stopping", "stopped"),
                    "Explicitly stopped instance cannot auto-resume")
            require(self.resume or previous_status in (None, "ready_to_start"),
                    "Existing instance requires --resume; refusing duplicate launch")
            signal.signal(signal.SIGTERM, self.signal)
            signal.signal(signal.SIGINT, self.signal)
            self.status(status="starting", stop_reason=None, completed_stages=[])
            previous = None
            known_results = {}
            stages = list(self.manifest["stages"])
            try:
                for index, stage in enumerate(stages):
                    for path, expected_hash in self.manifest.get("code_sha256", {}).items():
                        require(digest(self.root / path) == expected_hash,
                                f"Locked runtime code changed: {path}")
                    folder = self.root / stage["experiment_dir"]
                    no_live_stage_process(folder)
                    results_path = folder / "results.json"
                    if results_path.exists():
                        require(self.resume, "Completed stage already exists")
                        previous = read_json(results_path)
                        known_results[stage["version"]] = previous
                        require(digest(previous["retained_model"]["path"])
                                == previous["retained_model"]["sha256"], "Retained model changed")
                        complete_retention(folder)
                        self.record_completed(stage, folder, previous)
                        self.maybe_extend(stages, index, known_results)
                        continue
                    config = read_json(folder / "config.json")
                    binding_path = folder / "source-binding.json"
                    if not binding_path.exists():
                        binding_path = folder / "pool-binding.json"
                    allowed_hashes = {stage["prepared_config_sha256"]}
                    if binding_path.exists():
                        binding = read_json(binding_path)
                        allowed_hashes = {binding["config_sha256"]}
                        if binding.get("status") == "pending":
                            allowed_hashes.add(stage["prepared_config_sha256"])
                            allowed_hashes.add(binding.get("previous_config_sha256"))
                    require(digest(folder / "config.json") in allowed_hashes,
                            "Frozen stage configuration hash changed")
                    if (stage.get("rank_opponents", stage.get("conditional"))
                            and not (folder / "source-binding.json").exists()):
                        from .stage_extension import bind_ranked_pool, ranking_input

                        input_path = folder / "ranking-input.json"
                        if not input_path.exists():
                            inputs = ranking_input(self.manifest, previous, known_results)
                            write_json(input_path, inputs)
                        elif self.manifest.get("model_version_registry"):
                            validate_sealed_lifecycle_ranking(self.root, self.manifest,
                                previous, known_results, read_json(input_path))
                        else:
                            inputs = ranking_input(self.manifest, previous, known_results)
                            sealed_inputs = read_json(input_path)
                            require(all(sealed_inputs[key] == inputs[key] for key in
                                        ("protected_version", "candidates", "protocol")),
                                    "Ranking input changed during recovery")
                        if not (folder / "ranking-results.json").exists():
                            self.run_child(self.command("rank", folder), stage, index, "ranking")
                        config = bind_ranked_pool(self.root, folder, previous)
                    if index:
                        config = bind_source(self.root, folder, previous, known_results)
                    verify_config(config)
                    require(digest(self.root / config["initial_model"])
                            == config["initial_model_sha256"],
                            "Stage source hash mismatch")
                    source = inspect_checkpoint(self.root / config["initial_model"])
                    for spec in config["opponent_pool"]:
                        require(digest(self.root / spec["model"]) == spec["sha256"],
                                "Frozen opponent changed")
                    write_json(folder / "source-checkpoint.json", source)
                    models = self.root / "runs" / config["name"]
                    if models.exists():
                        require(self.resume, "Stage output exists; refusing duplicate training")
                        progress = read_json(folder / "progress.json")
                        if progress["status"] != "completed":
                            self.recover(stage, index, folder, config)
                    else:
                        self.train(stage, index, folder, config)
                    self.run_child(self.command("finalize", folder), stage, index, "finalizing")
                    previous = read_json(results_path)
                    known_results[stage["version"]] = previous
                    self.record_completed(stage, folder, previous)
                    self.status(phase="stage_completed", budget=self.budget(index, {}))
                    self.maybe_extend(stages, index, known_results)
                self.status(status="completed", phase="completed",
                            stop_reason=self.state.get("extension_stop_reason") or
                            "Final authorized stage and independent evaluations completed")
            except BaseException as exc:
                self.status(status="stopped" if self.stopping else "needs_attention",
                            error=f"{type(exc).__name__}: {exc}",
                            stop_reason=self.state.get("stop_reason") or
                            "Controller stopped; diagnosis required before recovery")
                raise

    def maybe_extend(self, stages, index, results):
        if index != len(self.manifest["stages"]) - 1 or not self.manifest.get("conditional_stages"):
            return
        from .stage_extension import extension_gate

        gate_path = self.folder / "extension-gate.json"
        gate = read_json(gate_path) if gate_path.exists() else extension_gate(
            self.root, self.manifest, results)
        if not gate_path.exists():
            write_json(gate_path, gate)
        if gate["eligible"]:
            stages.extend(self.manifest["conditional_stages"])
            self.state["conditional_extension_enabled"] = True
            self.status(conditional_extension_enabled=True, extension_gate=str(gate_path),
                        budget=self.budget(index, {}))
        else:
            self.status(conditional_extension_enabled=False, extension_gate=str(gate_path),
                        extension_stop_reason=gate["reason"])

    def record_completed(self, stage, folder, result):
        require(result["status"] == "completed", "Incomplete stage result")
        self.state.setdefault("completed_stages", []).append({"version": stage["version"],
            "ppo_iterations": 100, "samples": result["fresh_samples"],
            "adam_updates": result["actual_adam_updates"],
            "results_path": str(folder / "results.json"),
            "selected_model": result["retained_model"], "selection": result["selection"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recovery-reason")
    parser.add_argument("--worker", choices=("finalize", "repair", "rank"))
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        manifest = read_json(args.manifest)
        if args.worker == "rank":
            from .stage_extension import rank_candidates

            worker = rank_candidates
        else:
            worker = finalize if args.worker == "finalize" else repair_boundary
        worker(Path(manifest["root"]), args.experiment_dir, "cuda")
    else:
        StageChain(args.manifest, args.resume, args.recovery_reason).run()


if __name__ == "__main__":
    main()
