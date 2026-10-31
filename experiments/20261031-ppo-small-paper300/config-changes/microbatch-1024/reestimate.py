"""Recompute the pending 300-round budget from retained measurements; no training."""

import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    folder = Path(__file__).resolve().parent
    experiment = folder.parents[1]
    repository = experiment.parent.parent
    amp = experiment.parent / "20261031-bf16-mixed-precision"
    config = json.loads((experiment / "config.json").read_text())
    progress_path = experiment / "progress.json"
    if not progress_path.exists():
        progress_path = experiment / "experiment.json"
    progress = json.loads(progress_path.read_text())
    metrics_path = experiment / "metrics.jsonl"
    pauses_path = experiment / "profiling-pauses.jsonl"
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    pauses = [json.loads(line) for line in pauses_path.read_text().splitlines() if line.strip()]
    sampling = mean(row["rollout_seconds"] for row in metrics)
    update = (sum(row["update_seconds"] for row in metrics) - sum(row["seconds"] for row in pauses)) / len(metrics)
    remaining = config["iterations"] - progress["completed_iterations"]
    per_round = config["decisions_per_iteration"]
    micro = config["micro_batch_size"]
    inputs = [metrics_path, pauses_path, progress_path]
    estimates = []
    # Both measurement orders are retained to expose the spread of the extrapolation.
    for relative in ("results.json", "reverse-repeat/results.json"):
        path = amp / relative
        inputs.append(path)
        data = json.loads(path.read_text())
        assert data["status"] == "completed"
        rows = {(row["microbatch"], row["precision"]): row for row in data["rows"]}
        baseline_ms = rows[256, "fp32"]["effective_batch_update"]["wall_ms"]["median"]
        inference_ms = data["sampling_inference"]["fp32"]["timing"]["wall_ms"]["median"]
        for mode in ("fp32", "bf16"):
            target_ms = rows[micro, mode]["effective_batch_update"]["wall_ms"]["median"]
            target_inference_ms = data["sampling_inference"][mode]["timing"]["wall_ms"]["median"]
            new_sampling = sampling + (target_inference_ms - inference_ms) * config["n_steps"] / 1000
            new_update = update * target_ms / baseline_ms
            seconds = new_sampling + new_update
            estimates.append({
                "mode": "fp32_tf32_allowed" if mode == "fp32" else "bf16_mixed_fp32",
                "source": str(path.relative_to(repository)), "micro_batch_size": micro,
                "sampling_seconds_per_iteration": new_sampling,
                "update_seconds_per_iteration": new_update,
                "seconds_per_iteration": seconds,
                "total_300_rounds_hours": seconds * config["iterations"] / 3600,
                "remaining_297_rounds_hours": seconds * remaining / 3600,
            })
    scenarios = {}
    for mode in ("fp32_tf32_allowed", "bf16_mixed_fp32"):
        choices = [row for row in estimates if row["mode"] == mode]
        scenarios[mode] = {
            "main_estimate": choices[-1],
            "total_hours_range_across_measurement_orders": sorted(row["total_300_rounds_hours"] for row in choices),
            "remaining_hours_range_across_measurement_orders": sorted(row["remaining_297_rounds_hours"] for row in choices),
            "enabled_in_formal_training": mode == "fp32_tf32_allowed",
        }
    main_estimate = scenarios["fp32_tf32_allowed"]["main_estimate"]
    result = {
        "target_iterations": config["iterations"],
        "completed_iterations": progress["completed_iterations"], "remaining_iterations": remaining,
        "micro_batch_size": micro, "effective_batch_size": config["batch_size"],
        "gradient_accumulation": config["batch_size"] // micro,
        "samples_per_iteration": per_round, "target_samples": per_round * config["iterations"],
        "remaining_samples": per_round * remaining,
        "max_optimizer_steps_per_iteration": per_round // config["batch_size"],
        "max_optimizer_steps": per_round * config["iterations"] // config["batch_size"],
        "microbatches_per_iteration": per_round // micro,
        "max_microbatches": per_round * config["iterations"] // micro,
        "microbatches_if_resuming_after_three_micro256_iterations":
            per_round * progress["completed_iterations"] // 256 + per_round * remaining // micro,
        "estimated_sampling_seconds_per_iteration": main_estimate["sampling_seconds_per_iteration"],
        "estimated_update_seconds_per_iteration": main_estimate["update_seconds_per_iteration"],
        "estimated_total_hours": main_estimate["total_300_rounds_hours"],
        "estimated_remaining_hours": main_estimate["remaining_297_rounds_hours"],
        "observation_cache_gib": 12.9140625, "complete_training_cache_gib": 13.0703125,
        "cumulative_observation_tib": 3.783416748046875,
        "precision_scenarios": scenarios, "all_estimates": estimates,
        "basis": {
            "sampling_seconds_mean_three_completed_rounds": sampling,
            "update_seconds_mean_pause_corrected": update,
            "pause_seconds_removed": sum(row["seconds"] for row in pauses),
            "update_formula": "historical micro256 full-round update seconds * same-run micro1024 update ms / same-run micro256 FP32-with-TF32-allowed update ms",
            "sampling_formula": "historical sampling seconds + (target minus baseline no-grad inference ms) * 16384 / 1000",
            "input_sha256": {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        },
        "limitations": [
            "These are extrapolations, not full-round micro1024 timings or confidence intervals.",
            "BF16 is an evaluated alternative, not enabled in the formal training entry point.",
            "Fixed-batch benchmarks exclude CPU decoding/H2D; scaling the whole update phase is approximate.",
            "Estimates exclude evaluation, checkpoint saves and interruptions; future game lengths can change.",
            "Microbatch counts assume no KL early stopping; max_microbatches assumes all 300 rounds use 1024.",
        ],
        "status": "estimate_only; formal training stopped by user",
    }
    (folder / "estimate.json").write_text(json.dumps(result, indent=2) + "\n")
    (experiment / "budget-estimate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key in ("micro_batch_size", "gradient_accumulation", "max_microbatches", "estimated_total_hours", "estimated_remaining_hours", "precision_scenarios")}, indent=2))


if __name__ == "__main__":
    main()
