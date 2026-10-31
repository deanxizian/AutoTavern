"""Paired full-game FP32/BF16 evaluation of the unchanged experiment-01 best."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import taverngym as tg
from taverngym.learning.experiment import load_frozen_policy, now, write_json
from taverngym.learning.model import ModelOutput
from taverngym.learning.tracking import WandbTracker
from taverngym.learning.vector_env import NativeVectorEnv


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
FOLDER = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip"
CONDITIONS = ("sampled_fp32", "sampled_bf16", "greedy_fp32", "greedy_bf16")
METRICS = ("cycle_returns", "actions_per_game", "reversible_actions", "cap_end_rate")


class PrecisionPolicy:
    def __init__(self, policy, precision):
        self.policy, self.precision = policy, precision
        self.raw_logits_dtype = None

    def predict(self, obs, deterministic=False):
        tensors, _ = self.policy.obs_to_tensor(obs)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.precision == "bf16" else nullcontext()
        with torch.no_grad():
            with context:
                raw = self.policy.network(tensors, mask_actions=False)
            self.raw_logits_dtype = str(raw.logits.dtype)
            output = ModelOutput(*(value.float() for value in raw))
            distribution = self.policy._distribution(output, tensors)
            actions = distribution.get_actions(deterministic=deterministic)
        return actions.cpu().numpy(), None


def statistics(rows):
    placements = np.array([row["placement"] for row in rows])
    return {"games": len(rows), "mean_placement": float(placements.mean()),
            "top4_rate": float((placements <= 4).mean()), "first_rate": float((placements == 1).mean()),
            **{name: float(np.mean([row[name] for row in rows])) for name in METRICS}}


def play_block(policy, condition, block, config, stream):
    seed = config["seeds"][block]
    torch.manual_seed(seed)
    env = NativeVectorEnv(64, seed=seed, threads=4, config=config["simulator_config"],
                          reward_shaping={"operation_cost": 0.0, "cycle_cost": 0.0}, rotate_seats=True)
    obs = env.reset()
    completed = np.zeros(64, dtype=bool)
    rows = []
    started, last_write = time.monotonic(), time.monotonic()
    vector_steps = 0
    try:
        while not completed.all():
            actions, _ = policy.predict(obs, deterministic=condition.startswith("greedy"))
            obs, _, dones, infos = env.step(actions)
            vector_steps += 1
            for i in np.flatnonzero(dones & ~completed):
                info = infos[i]
                assert 1 <= info["placement"] <= 8
                row = {"condition": condition, "block": block, "seed": seed, "lane": int(i),
                       "episode_index": 0, "placement": info["placement"],
                       **{name: info["reward_components"][name] for name in METRICS}}
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                completed[i] = True
            if time.monotonic() - last_write >= 5:
                write_json(FOLDER / "progress.json", {"status": "running", "updated_at": now(),
                           "condition": condition, "block": block, "games_finished_in_block": len(rows),
                           "target_games_in_block": 64, "vector_steps": vector_steps,
                           "block_elapsed_seconds": time.monotonic() - started})
                last_write = time.monotonic()
            if vector_steps > 10_000:
                raise RuntimeError("Evaluation block exceeded its bounded decision budget")
    finally:
        env.close()
    return rows, {"condition": condition, "block": block, "seconds": time.monotonic() - started,
                  "vector_steps": vector_steps, "raw_logits_dtype": policy.raw_logits_dtype}


def paired_comparison(rows, mode):
    matched = {}
    for precision in ("fp32", "bf16"):
        selected = sorted(rows[f"{mode}_{precision}"], key=lambda r: (r["block"], r["lane"]))
        matched[precision] = selected
    keys = lambda values: [(r["block"], r["seed"], r["lane"], r["episode_index"]) for r in values]
    assert keys(matched["fp32"]) == keys(matched["bf16"]) and len(matched["fp32"]) == 256
    rng = np.random.default_rng(8675309)
    indices = rng.integers(0, 256, size=(10_000, 256))
    fp32 = np.array([r["placement"] for r in matched["fp32"]])
    bf16 = np.array([r["placement"] for r in matched["bf16"]])
    deltas = {"mean_placement": bf16 - fp32,
              "top4_percentage_points": 100 * ((bf16 <= 4).astype(float) - (fp32 <= 4)),
              "first_percentage_points": 100 * ((bf16 == 1).astype(float) - (fp32 == 1))}
    result = {"paired_games": 256, "difference_direction": "BF16 minus FP32",
              "interval_method": "10,000 paired game bootstrap resamples, percentile 95% CI"}
    for name, differences in deltas.items():
        means = differences[indices].mean(axis=1)
        result[name] = {"delta": float(differences.mean()),
                        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])]}
    result["games_with_same_placement"] = int((fp32 == bf16).sum())
    return result


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    assert torch.cuda.is_bf16_supported() and tg.build_is_current()
    config = json.loads((FOLDER / "config.json").read_text())
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert source_hash == config["source_sha256"]
    if (FOLDER / "games.jsonl").exists():
        raise FileExistsError("Refusing to overwrite an existing evaluation")
    policy = load_frozen_policy(SOURCE, "cuda")
    wrappers = {precision: PrecisionPolicy(policy, precision) for precision in ("fp32", "bf16")}
    smoke = NativeVectorEnv(4, seed=123)
    obs = smoke.reset()
    for deterministic in (False, True):
        torch.manual_seed(123)
        expected = policy.predict(obs, deterministic=deterministic)[0]
        torch.manual_seed(123)
        actual = wrappers["fp32"].predict(obs, deterministic=deterministic)[0]
        np.testing.assert_array_equal(expected, actual)
    smoke.close()
    rows = {condition: [] for condition in CONDITIONS}
    timings = []
    tracker = WandbTracker(FOLDER, config, kind="bf16-game-evaluation")
    tracker.run.define_metric("evaluated_games")
    tracker.run.define_metric("test/*", step_metric="evaluated_games")
    started = time.monotonic()
    metadata = {"status": "running", "started_at": now(), "source_model": str(SOURCE.relative_to(ROOT)),
                "source_sha256": source_hash, "config": "config.json", "weights_uploaded": False,
                "training_started": False, "second_training_restarted": False,
                "native_build": tg.build_info(), "fp32_wrapper_matches_original": True}
    write_json(FOLDER / "experiment.json", metadata)
    exit_code = 1
    try:
        with (FOLDER / "games.jsonl").open("x", buffering=1) as stream:
            for block in range(4):
                # Rotate condition order to balance timing drift; scores depend
                # only on fixed game/decision seeds, not scheduling or wall time.
                order = CONDITIONS[block:] + CONDITIONS[:block]
                for condition in order:
                    precision = condition.rsplit("_", 1)[1]
                    new_rows, timing = play_block(wrappers[precision], condition, block, config, stream)
                    rows[condition].extend(new_rows)
                    timings.append(timing)
                    summary = statistics(rows[condition])
                    write_json(FOLDER / f"{condition}.json", {**summary, "games_detail": rows[condition]})
                    tracker.log({"evaluated_games": summary["games"],
                                 **{f"test/{condition}_{k}": v for k, v in summary.items()}}, 0)
                    print(condition, f"block={block}", summary, flush=True)
        assert all(len(values) == 256 for values in rows.values())
        assert wrappers["bf16"].raw_logits_dtype == "torch.bfloat16"
        assert {str(p.dtype) for p in policy.parameters()} == {"torch.float32"}
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
        results = {"status": "completed", "completed_at": now(), "source_sha256": source_hash,
                   "conditions": {name: statistics(values) for name, values in rows.items()},
                   "paired": {mode: paired_comparison(rows, mode) for mode in ("sampled", "greedy")},
                   "timings": timings, "elapsed_seconds": time.monotonic() - started,
                   "total_scored_games": 1024, "weights_unchanged": True,
                   "formal_training_restarted": False, "new_checkpoints_saved": False,
                   "float_weights_dtype": "float32", "bf16_logits_dtype": wrappers["bf16"].raw_logits_dtype,
                   "probability_dtype": "float32", "fp32_wrapper_parity_verified": True,
                   "limitations": ["256 paired games per mode; CIs describe this evaluation, not a universal strength guarantee.",
                       "Same simulator and policy seeds do not force identical trajectories when numerical precision changes actions.",
                       "Auto-reset games after a lane's first scored episode are ignored; pairing uses block, seed, lane and episode index."]}
        write_json(FOLDER / "results.json", results)
        values = {"evaluated_games": 256}
        for mode, comparison in results["paired"].items():
            for metric in ("mean_placement", "top4_percentage_points", "first_percentage_points"):
                values[f"test/{mode}_{metric}_delta_bf16_minus_fp32"] = comparison[metric]["delta"]
        tracker.log(values, 0)
        metadata.update(status="completed", completed_at=now(), results="results.json", scored_games=1024)
        write_json(FOLDER / "experiment.json", metadata)
        write_json(FOLDER / "progress.json", {"status": "completed", "completed_at": now(), "scored_games": 1024})
        exit_code = 0
        print(json.dumps({"conditions": results["conditions"], "paired": results["paired"]}, indent=2), flush=True)
    finally:
        tracker.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
