"""ABBA short sampling comparison of one versus two batched frozen V1 seats."""

import importlib.util
import json
from pathlib import Path
import statistics
import traceback

import torch

from taverngym.learning.actor_sampling import PolicyActor, UnifiedActorSampler
from taverngym.learning.experiment import load_frozen_policy, now, write_json


FOLDER = Path(__file__).resolve().parent
HELPER = FOLDER.parent / "20261031-unified-actor-sampling/run.py"
spec = importlib.util.spec_from_file_location("actor_sampling_probe", HELPER)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
ROOT = probe.ROOT


def main():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    protected = {str(p.relative_to(ROOT)): probe.sha(p) for p in (probe.SOURCE, probe.STOPPED)}
    assert json.loads(probe.FORMAL.read_text())["status"] == "stopped_by_user"
    policies = {name: load_frozen_policy(probe.SOURCE, "cuda", forward_precision="bf16")
                for name in ("learner", "bot_v1")}
    versions = {name: [p._version for p in policy.parameters()] for name, policy in policies.items()}
    raw = {name: set() for name in policies}
    hooks = []
    for name, policy in policies.items():
        assert {p.dtype for p in policy.parameters()} == {torch.float32}
        assert not any(p.requires_grad for p in policy.parameters())
        hooks.append(policy.network.pointer[-1].register_forward_hook(
            lambda m, a, out, identity=name: raw[identity].add(str(out.dtype))))
    warm = UnifiedActorSampler(256, PolicyActor(policies["learner"]), PolicyActor(policies["bot_v1"]),
                               frozen_opponents=2, seed=43, threads=4, config={"turn_action_caps": [60]})
    pool = {k: v[warm.observations["ready"]] for k, v in warm.observations.items() if k != "ready"}
    for policy in policies.values():
        actor = PolicyActor(policy)
        for size in (2, 32, 256, 512):
            actor.act({k: v[:size] for k, v in pool.items()})
    del warm, pool
    rows, sampler = [], None
    try:
        for count, seed in ((1, 43), (2, 43), (2, 44), (1, 44)):
            torch.manual_seed(1000 + seed)
            learner, fixed = (PolicyActor(policies[name]) for name in ("learner", "bot_v1"))
            audit = probe.Audit(256)
            sampler = UnifiedActorSampler(256, learner, fixed, frozen_opponents=count, seed=seed,
                threads=4, config={"turn_action_caps": [60]},
                reward_shaping={"operation_cost": 0.0001, "cycle_cost": 0.0002})
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            write_json(FOLDER / "progress.json", {"status": "running", "frozen_opponents": count,
                       "seed": seed, "updated_at": now(), "completed_cases": len(rows)})
            row = sampler.collect(128, audit)
            assert (audit.counts == 128).all() and sum(learner.batch_sizes) == 32768
            row.update({"frozen_opponents": count, "scripted_opponents": 7 - count, "seed": seed,
                "learner": probe.actor_stats(learner), "bot_v1": probe.actor_stats(fixed),
                "audit_passed": True, "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30})
            rows.append(row)
            write_json(FOLDER / "partial-results.json", rows)
            print(json.dumps({"fixed_v1": count, "seed": seed, "seconds": row["sampling_seconds"],
                "samples_per_second": row["learner_samples_per_second"],
                "bot_median_batch": row["bot_v1"]["batch_median"]}), flush=True)
            sampler = None
        summary = {}
        bf16 = json.loads((FOLDER.parent / "20261031-bf16-mixed-precision/summary.json").read_text())
        update_seconds = 32 * next(x for x in bf16["rows"] if x["microbatch"] == 1024)["bf16_update_wall_ms"] / 1000
        for count in (1, 2):
            selected = [r for r in rows if r["frozen_opponents"] == count]
            elapsed = sum(r["sampling_seconds"] for r in selected)
            sps = 65536 / elapsed
            batches = [b for r in selected for b in r["bot_v1"]["batch_sizes"]]
            summary[str(count)] = {"learner_samples": 65536, "sampling_seconds": elapsed,
                "learner_samples_per_second": sps, "bot_batch_median": statistics.median(batches),
                "bot_batch_mean": statistics.mean(batches), "bot_batch_max": max(batches),
                "bot_forward_calls": len(batches), "bot_decisions": sum(batches),
                "extrapolated_ppo_sampling_seconds": 262144 / sps,
                "extrapolated_100ppo_sampling_minutes": 26214400 / sps / 60,
                "extrapolated_ppo_update_seconds_gpu_only": update_seconds,
                "extrapolated_100ppo_sampling_plus_updates_minutes": (262144 / sps + update_seconds) * 100 / 60}
        for name, policy in policies.items():
            assert raw[name] == {"torch.bfloat16"}
            assert versions[name] == [p._version for p in policy.parameters()]
            assert not policy.optimizer.state and all(p.grad is None for p in policy.parameters())
        assert protected == {str(p.relative_to(ROOT)): probe.sha(p) for p in (probe.SOURCE, probe.STOPPED)}
        assert json.loads(probe.FORMAL.read_text())["status"] == "stopped_by_user"
        result = {"status": "completed", "completed_at": now(), "device": torch.cuda.get_device_name(),
            "games": 256, "steps_per_game_per_case": 128, "threads": 4, "torch_threads": 2,
            "policy_instances": 2, "fixed_seats_share_one_policy": True, "precision": "bf16 forward, fp32 weights and outputs",
            "raw_head_dtypes": {k: sorted(v) for k, v in raw.items()}, "optimizer_updates": 0, "models_saved": 0,
            "formal_training_still_stopped": True, "protected_model_hashes": protected,
            "summary": summary, "two_vs_one_sampling_time_ratio": summary["2"]["sampling_seconds"] / summary["1"]["sampling_seconds"],
            "cases": rows, "source_hashes": {str(p.relative_to(ROOT)): probe.sha(p) for p in
                (Path(__file__), HELPER, ROOT / "python/taverngym/learning/actor_sampling.py")},
            "limitations": ["128-step short sampling probe, not a complete 1024-step PPO rollout or training run.",
                "Different lobby mixes and scheduling change trajectories; this is not a strength evaluation.",
                "Training time extrapolation excludes rollout-buffer writes, GAE, evaluation, saving, and TavernBench.",
                "GPU update estimate reuses the prior 8192-effective-batch BF16 benchmark with microbatch 1024."]}
        write_json(FOLDER / "results.json", result)
        write_json(FOLDER / "progress.json", {"status": "completed", "updated_at": now()})
        print(json.dumps({"summary": summary, "two_vs_one_sampling_time_ratio": result["two_vs_one_sampling_time_ratio"]}), flush=True)
    except Exception:
        write_json(FOLDER / "failure.json", {"at": now(), "traceback": traceback.format_exc(),
                   "native_failure": sampler.last_failure if sampler is not None else None, "completed_cases": rows})
        raise
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    main()
