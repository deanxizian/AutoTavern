"""The authorized V6 gate and bounded development selection of historical bots."""

import hashlib
import json
import math
from pathlib import Path
import shutil

from .stage_chain import digest, inspect_checkpoint, json_digest, read_json, require
from .experiment import now, write_json
from .rewards import reward_protocols_equal


def extension_gate(root, manifest, results):
    checks = {}
    for version, stage in zip(("v4", "v5", "v6"), manifest["stages"]):
        folder = root / stage["experiment_dir"]
        result = results[version]
        checks[f"{version}_complete"] = (
            result["status"] == "completed" and result["fresh_samples"] == 13107200
            and read_json(folder / "retention.json")["status"] == "completed")
        checks[f"{version}_finite_and_consistent"] = result.get("verification", {}).get(
            "finite_and_protocol_consistent", False)
        checks[f"{version}_no_unresolved_native_failure"] = not (
            folder / "native-action-failure.json").exists()
    v6 = results["v6"]
    folder = root / manifest["stages"][-1]["experiment_dir"]
    selected = read_json(folder / f"evaluation-{v6['selection']['ppo_iteration']:04d}.json")
    # This is only a spending gate. It never changes stage best or consults test.
    values = [selected.get(name) for name in ("cycle_forfeit_rate", "action_limit_forfeit_rate")]
    valid = all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                for value in values)
    rate = sum(values) if valid else None
    valid = valid and rate <= 1 + 1e-12
    checks["v6_self_forfeit_metrics_valid"] = valid
    if manifest.get("cycle_rule") == "constant_penalty_forced_end_no_forfeits":
        # Zero is a protocol-integrity check, never evidence of training improvement.
        checks["v6_disabled_early_termination_rates_are_zero"] = valid and values == [0, 0]
    elif manifest.get("cycle_rule") == "constant_penalty_no_cycle_forfeit":
        checks["v6_cycle_forfeit_disabled_and_zero"] = valid and values[0] == 0
        checks["v6_selected_best_not_all_action_limit_forfeits"] = (
            valid and values[1] < 1 - 1e-12)
    else:
        checks["v6_selected_best_not_all_self_forfeits"] = valid and rate < 1 - 1e-12
    if manifest.get("stop_extension_after_three_step0_bests"):
        checks["not_three_consecutive_stage_best_step0"] = not all(
            results[version]["selection"]["ppo_iteration"] == 0
            for version in ("v4", "v5", "v6"))
    return {"checked_at": now(), "eligible": all(checks.values()), "checks": checks,
            "v6_selected_validation_self_forfeit_rate": rate,
            "used_for_model_selection": False, "heldout_used_for_gate": False,
            "reason": "all primary checks passed" if all(checks.values()) else
                      "unresolved primary-stage problem; stop after V6"}


def ranking_input(manifest, previous, known_results=None):
    if manifest.get("pool_selection") == "top3":
        root = Path(manifest["root"])
        registry = {version: dict(spec) for version, spec in manifest["historical_models"].items()}
        known = {previous["model_version"]: previous, **(known_results or {})}
        for version, result in known.items():
            model = result["retained_model"]
            registry[version] = {"name": f"bot_{version}", "count": 2,
                "model": str(Path(model["path"]).relative_to(root)), "sha256": model["sha256"],
                "model_version": version, "source_stage": version,
                "reward_shaping": result["reward_shaping"], "protocol_source": result["config_path"]}
        order = manifest["ranking_protocol"]["version_order"]
        prefix = order[:order.index(previous["model_version"]) + 1]
        lifecycle = {}
        if manifest.get("model_version_registry"):
            path = root / manifest["model_version_registry"]
            # Capture one byte sequence for both parse and hash, so a concurrent
            # registry edit cannot produce a mixed snapshot. Never reread later.
            payload = path.read_bytes()
            document = json.loads(payload)
            require(document.get("schema_version") == 1, "Unsupported model version registry")
            versions = document["versions"]
            require(all(version in versions for version in prefix),
                    "Model version registry is missing a historical version")
            excluded = [version for version, info in versions.items()
                        if info.get("status") == "retired"]
            require(set(manifest.get("excluded_opponent_versions", [])) <= set(excluded),
                    "Excluded opponents must be retired in the model version registry")
            lifecycle = {"model_version_registry": manifest["model_version_registry"],
                "model_version_registry_sha256": hashlib.sha256(payload).hexdigest(),
                "model_version_registry_snapshot": document,
                "excluded_opponent_versions": excluded}
            expected_versions = [version for version in prefix if version not in excluded]
            require(all(registry[version]["model_version"] == version
                        for version in expected_versions if version in registry),
                    "Historical model registry keys cannot alias other versions")
            candidates = [registry[version] for version in expected_versions if version in registry]
            expected = len(expected_versions)
        else:
            candidates = [registry[version] for version in order if version in registry]
            expected = len(prefix)
        require(len(candidates) == expected and (3 if lifecycle else 4) <= expected <= 8,
                "All available historical versions must participate in top-three selection")
        result = {"created_at": now(), "pool_selection": "top3", "protected_version": None,
                "learning_source_version": previous["model_version"],
                "candidate_count": len(candidates), "candidates": candidates,
                "protocol": manifest["ranking_protocol"], **lifecycle}
        validate_lifecycle_input(result)
        return result
    candidates = previous["opponent_pool"]
    require(len(candidates) == 5, "Expected exactly five removable retained versions")
    latest = previous["model_version"]
    require(all(spec["model_version"] != latest for spec in candidates),
            "Latest protected generation cannot be a removal candidate")
    return {"created_at": now(), "protected_version": latest,
            "candidates": candidates, "protocol": manifest["ranking_protocol"]}


def validate_lifecycle_input(inputs):
    """Validate the sealed snapshot, never the mutable global registry file."""
    if not inputs.get("model_version_registry"):
        return
    document = inputs["model_version_registry_snapshot"]
    require(document.get("schema_version") == 1, "Unsupported model version registry")
    versions = document["versions"]
    require(all(info.get("status") in ("active", "retired")
                and (info["status"] == "retired" or info.get("canonical_version") == version)
                for version, info in versions.items()),
            "Active model versions must be canonical, never aliases")
    excluded = set(inputs["excluded_opponent_versions"])
    require(excluded == {v for v, info in versions.items() if info["status"] == "retired"},
            "Sealed excluded opponents disagree with lifecycle snapshot")
    specs = [*inputs["candidates"], *inputs["protocol"]["reference_pool"]]
    candidate_versions = [spec["model_version"] for spec in inputs["candidates"]]
    require(len(set(candidate_versions)) == len(candidate_versions) == inputs["candidate_count"],
            "Lifecycle candidate versions must be unique and complete")
    require(all(spec["model_version"] in versions
                and versions[spec["model_version"]]["status"] == "active"
                and spec["model_version"] not in excluded for spec in specs),
            "Retired or unregistered version in candidates/reference pool")


def rank_candidates(root, folder, device):
    import torch
    from .experiment import evaluate, load_frozen_opponents
    from .ppo import EntityPPO, PlacementBuffer
    from .precision import configure_precision

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    inputs = read_json(folder / "ranking-input.json")
    validate_lifecycle_input(inputs)
    input_hash = digest(folder / "ranking-input.json")
    protocol = inputs["protocol"]
    require(protocol["games"] == 64 and protocol["seed"] == 60000007,
            "Ranking protocol changed")
    reference = {"opponent_pool": protocol["reference_pool"], "frozen_opponents": 6,
                 "forward_precision": "bf16", "opponent_forward_precision": "bf16"}
    opponents = load_frozen_opponents(reference, device)
    rows = []
    for spec in inputs["candidates"]:
        path = root / spec["model"]
        require(digest(path) == spec["sha256"], "Ranking candidate checkpoint changed")
        checkpoint = inspect_checkpoint(path)
        candidate_protocol = checkpoint["reward_shaping"]
        if spec["model_version"] in ("v1", "v2", "v3"):
            require(not candidate_protocol or (
                candidate_protocol.get("cycle_forfeit_repetitions") is None
                and candidate_protocol.get("turn_action_forfeit_limit") is None),
                "Historical generation unexpectedly has new termination rules")
        else:
            require(reward_protocols_equal(candidate_protocol, spec["reward_shaping"]),
                    "Ranking must use the candidate's actual generation protocol")
        output = folder / f"ranking-{spec['model_version']}.json"
        if output.exists():
            record = read_json(output)
            result = record["result"]
            require(record.get("input_sha256") == input_hash
                    and record["checkpoint"] == checkpoint
                    and record["candidate_protocol"] == candidate_protocol
                    and result["games"] == 64 and result["seed"] == 60000007
                    and len(result["placements"]) == 64
                    and result["mean_placement"] == sum(result["placements"]) / 64,
                    "Sealed ranking result differs from the locked input/protocol")
            rows.append(record)
            continue
        model = EntityPPO.load(path, device=device, n_envs=1, n_steps=2, batch_size=2,
                              rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
        configure_precision(model, reference, opponents)
        result = evaluate(model, protocol["games"], protocol["seed"],
            simulator_config={"turn_action_caps": [60]}, reward_shaping=candidate_protocol,
            opponent_policy=opponents, frozen_opponents=6, rotate_seats=True)
        require(math.isfinite(result["mean_placement"]) and len(result["placements"]) == 64,
                "Invalid development ranking result")
        record = {"model_version": spec["model_version"], "checkpoint": checkpoint,
                  "input_sha256": input_hash,
                  "candidate_protocol": candidate_protocol, "result": result,
                  "used_for_stage_best_selection": False, "heldout_used": False}
        write_json(output, record)
        rows.append(record)
        del model
    order = protocol["version_order"]
    if inputs.get("pool_selection") == "top3":
        require(len(rows) == inputs["candidate_count"]
                and (3 if inputs.get("model_version_registry") else 4) <= len(rows) <= 8,
                "Top-three ranking has missing or excessive candidates")
        ranked = sorted(rows, key=lambda item: (item["result"]["mean_placement"],
                                              -order.index(item["model_version"])))
        write_json(folder / "ranking-results.json", {"completed_at": now(), "status": "completed",
            "input_sha256": input_hash, "pool_selection": "top3", "protocol": protocol,
            "candidates": rows, "ranked_versions": [item["model_version"] for item in ranked],
            "selected_versions": [item["model_version"] for item in ranked[:3]],
            "protected_version": None, "heldout_used": False,
            "rule": "lowest stochastic mean placement; newer version first on pool-selection ties"})
        return
    removed = max(rows, key=lambda row: (row["result"]["mean_placement"],
                                       -order.index(row["model_version"])))
    write_json(folder / "ranking-results.json", {"completed_at": now(), "status": "completed",
        "input_sha256": input_hash,
        "protocol": protocol, "candidates": rows, "removed_version": removed["model_version"],
        "protected_version": inputs["protected_version"], "heldout_used": False,
        "rule": "remove highest stochastic mean placement; remove older version on ties"})


def bind_ranked_pool(root, folder, previous):
    ranking = read_json(folder / "ranking-results.json")
    require(ranking["status"] == "completed", "Ranking incomplete")
    config = read_json(folder / "config.json")
    if ranking.get("pool_selection") == "top3":
        inputs = read_json(folder / "ranking-input.json")
        validate_lifecycle_input(inputs)
        require(ranking["input_sha256"] == digest(folder / "ranking-input.json"),
                "Top-three pool differs from its sealed ranking input")
        versions = ranking["selected_versions"]
        require(len(set(versions)) == len(versions) == 3, "Top-three pool requires three versions")
        require(not set(versions) & set(inputs.get("excluded_opponent_versions", [])),
                "Retired version cannot occupy a bound opponent seat")
        registry = {spec["model_version"]: spec for spec in inputs["candidates"]}
        retained = [dict(registry[version], count=2) for version in versions]
    else:
        retained = [dict(spec, count=1) for spec in previous["opponent_pool"]
                    if spec["model_version"] != ranking["removed_version"]]
        latest = previous["retained_model"]
        retained.append({"name": f"bot_{previous['model_version']}", "count": 2,
            "model": str(Path(latest["path"]).relative_to(root)), "sha256": latest["sha256"],
            "model_version": previous["model_version"], "source_stage": previous["model_version"],
            "reward_shaping": previous["reward_shaping"], "protocol_source": previous["config_path"]})
    require(len(retained) == (3 if ranking.get("pool_selection") == "top3" else 5)
            and sum(spec["count"] for spec in retained) == 6,
            "Bound ranked pool has the wrong seat budget")
    history = folder / "preparation-history" / "before-pool-binding"
    history.mkdir(parents=True, exist_ok=True)
    if not (history / "config.json").exists():
        shutil.copy2(folder / "config.json", history / "config.json")
    original_hash = digest(history / "config.json")
    config.update(opponent_pool=retained, expected_opponent_counts={
        spec["name"]: spec["count"] for spec in retained}, config_status="pool_bound")
    evidence = {"status": "pending", "created_at": now(), "ranking": "ranking-results.json",
                "pool_selection": ranking.get("pool_selection", "remove_weakest"),
                "selected_versions": ranking.get("selected_versions"),
                "removed_version": ranking.get("removed_version"),
                "protected_version": ranking.get("protected_version"),
                "previous_config_sha256": original_hash, "config_sha256": json_digest(config)}
    write_json(folder / "pool-binding.json", evidence)
    write_json(folder / "config.json", config)
    evidence["status"] = "completed"
    write_json(folder / "pool-binding.json", evidence)
    return config
