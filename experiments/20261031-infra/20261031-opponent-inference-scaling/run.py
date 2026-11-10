"""Bounded 1-vs-4 opponent sampling and FP32/BF16 inference comparison.

The formal job stays running. These measurements share its GPU/CPU resources,
so report paired comparisons rather than claiming isolated peak throughput.
No optimizer updates or probe checkpoints are produced.
"""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from taverngym.learning.experiment import load_frozen_policy, now, write_json
from taverngym.learning.vector_env import MixedOpponentVectorEnv


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
FOLDER = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip"
FORMAL = ROOT / "experiments/20261031-exp02-mixed-opponents/20261031-ppo-mixed-v2"


class CountedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self.batches = []
        self.seconds = 0

    def predict(self, obs, deterministic=False):
        started = time.monotonic()
        result = self.policy.predict(obs, deterministic=deterministic)
        self.seconds += time.monotonic() - started
        self.batches.append(len(obs["features"]))
        return result


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    policy = load_frozen_policy(SOURCE, "cuda")
    assert {str(p.dtype) for p in policy.parameters()} == {"torch.float32"}
    assert torch.cuda.is_bf16_supported()
    cases, observations, rng, pools = {}, {}, {}, []
    for count in (1, 4):
        opponent, learner = CountedPolicy(policy), CountedPolicy(policy)
        env = MixedOpponentVectorEnv(256, seed=43, threads=4, opponent_policy=opponent,
                                    frozen_opponents=count, config={"turn_action_caps": [60]},
                                    reward_shaping={"operation_cost": 0.0001, "cycle_cost": 0.0002})
        observations[count] = env.reset()
        for _ in range(2):
            policy.predict(observations[count], deterministic=True)
        torch.manual_seed(1043)
        rng[count] = torch.cuda.get_rng_state()
        cases[count] = dict(env=env, opponent=opponent, learner=learner, blocks=[])
    try:
        for count in (1, 4, 4, 1):
            case = cases[count]
            torch.cuda.set_rng_state(rng[count])
            started = time.monotonic()
            for _ in range(32):
                actions, _ = case["learner"].predict(observations[count])
                observations[count], _, _, _ = case["env"].step(actions)
            seconds = time.monotonic() - started
            case["blocks"].append(seconds)
            rng[count] = torch.cuda.get_rng_state()
            pools.append({k: v.copy() for k, v in observations[count].items()})
            write_json(FOLDER / "progress.json", {"status": "running", "updated_at": now(),
                       "finished_block_opponents": count, "block_seconds": seconds})
            print(f"opponents={count}, 8192 learner samples: {seconds:.3f}s", flush=True)
        rows = []
        for count, case in cases.items():
            opponent, learner, env = (case[k] for k in ("opponent", "learner", "env"))
            wall = sum(case["blocks"])
            assert sum(opponent.batches) == env.opponent_decisions
            assert sum(learner.batches) == 16384
            rows.append({"frozen_opponents": count, "scripted_opponents": 7 - count,
                         "learner_samples": 16384, "sampling_seconds": wall, "learner_samples_per_second": 16384 / wall,
                         "learner_inference_seconds": learner.seconds, "opponent_inference_seconds": opponent.seconds,
                         "remaining_driver_seconds": wall - learner.seconds - opponent.seconds,
                         "opponent_decisions": env.opponent_decisions, "opponent_forward_calls": len(opponent.batches),
                         "opponent_mean_batch": statistics.mean(opponent.batches),
                         "opponent_median_batch": statistics.median(opponent.batches),
                         "opponent_max_batch": max(opponent.batches), "waiting_rounds": env.waiting_rounds,
                         "blocks_seconds": case["blocks"]})
        pool = {k: np.concatenate([obs[k] for obs in pools]) for k in pools[0]}
        inference = []
        with torch.inference_mode():
            for batch in (32, 256, 1024):
                tensors, _ = policy.obs_to_tensor({k: v[:batch] for k, v in pool.items()})
                cached = {}
                dtypes = {}

                def forward(precision):
                    context = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
                    with context:
                        raw = policy.network(tensors, mask_actions=False)
                    # Keep probability arithmetic FP32 in both cases.
                    logits = raw.logits.float().masked_fill(~tensors["action_mask"].bool(), -torch.inf)
                    return logits.softmax(-1), str(raw.logits.dtype)

                for precision in ("fp32_tf32_allowed", "bf16"):
                    for _ in range(2):
                        forward(precision)
                timings = {p: [] for p in ("fp32_tf32_allowed", "bf16")}
                for repeat in range(8):
                    order = ("fp32_tf32_allowed", "bf16") if repeat % 2 == 0 else ("bf16", "fp32_tf32_allowed")
                    for precision in order:
                        torch.cuda.synchronize()
                        started = time.monotonic()
                        probabilities, raw_dtype = forward(precision)
                        torch.cuda.synchronize()
                        timings[precision].append(1000 * (time.monotonic() - started))
                        cached[precision] = probabilities
                        dtypes[precision] = raw_dtype
                fp32, bf16 = cached["fp32_tf32_allowed"], cached["bf16"]
                assert torch.isfinite(fp32).all() and torch.isfinite(bf16).all()
                assert torch.allclose(bf16.sum(-1), torch.ones(batch, device="cuda"), atol=1e-6)
                inference.append({"batch": batch,
                    "fp32_median_ms": statistics.median(timings["fp32_tf32_allowed"]),
                    "bf16_median_ms": statistics.median(timings["bf16"]),
                    "speedup": statistics.median(timings["fp32_tf32_allowed"]) / statistics.median(timings["bf16"]),
                    "greedy_disagreement_fraction": (fp32.argmax(-1) != bf16.argmax(-1)).float().mean().item(),
                    "mean_probability_total_variation": ((fp32 - bf16).abs().sum(-1) / 2).mean().item(),
                    "max_probability_difference": (fp32 - bf16).abs().max().item(),
                    "bf16_raw_logits_dtype": dtypes["bf16"], "distribution_dtype": str(bf16.dtype),
                    "timings_ms": timings})
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
        result = {"status": "completed", "completed_at": now(), "device": torch.cuda.get_device_name(),
                  "source_sha256": source_hash, "weights_dtype": "float32", "formal_training_unchanged": True,
                  "formal_job_shared_resources": True, "no_probe_updates_or_checkpoints": True,
                  "sampling": rows, "inference": inference,
                  "formal_progress_at_end": json.loads((FORMAL / "progress.json").read_text()),
                  "limitations": ["Concurrent formal training means these are not isolated peak-speed measurements.",
                      "Sampling uses fixed V1 for the learner and opponents, 64 learner steps per environment, ABBA order.",
                      "BF16 timing covers GPU-resident neural forward plus FP32 masked softmax; not complete game throughput.",
                      "Numerical differences are not a win-rate validation."]}
        write_json(FOLDER / "results.json", result)
        print(json.dumps({k: result[k] for k in ("sampling", "inference")}, indent=2), flush=True)
    finally:
        for case in cases.values():
            case["env"].close()


if __name__ == "__main__":
    main()
