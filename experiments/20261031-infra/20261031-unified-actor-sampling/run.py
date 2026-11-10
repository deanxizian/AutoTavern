"""Bounded inference-only comparison: 6 scripts + fixed V1 + learner V1.

Two distinct policy instances use BF16 neural forwards and FP32 public outputs.
No optimizer, checkpoint writes, online tracking, or formal training are started.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

from taverngym.learning.actor_sampling import ActorTransition, PolicyActor, UnifiedActorSampler
from taverngym.learning.experiment import load_frozen_policy, now, write_json
from taverngym.learning.vector_env import MixedOpponentVectorEnv


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
FOLDER = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip"
STOPPED = ROOT / "runs/20261031-ppo-mixed-v2/latest_model.zip"
FORMAL = ROOT / "experiments/20261031-exp02-mixed-opponents/20261031-ppo-mixed-v2/progress.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Audit:
    """Check transition ownership/order without retaining the rollout in RAM."""

    def __init__(self, games):
        self.counts = np.zeros(games, np.int64)
        self.episodes = np.zeros(games, np.int64)
        self.episode_steps = np.zeros(games, np.int64)
        self.terminals = 0

    def __call__(self, transition):
        game = transition.game
        assert transition.rollout_step == self.counts[game]
        assert transition.episode == self.episodes[game]
        assert transition.episode_step == self.episode_steps[game]
        assert transition.observation["action_mask"][transition.action]
        assert transition.next_observation["action_mask"].any()
        assert np.isfinite([transition.value, transition.log_prob, transition.reward]).all()
        assert all(isinstance(x, np.float32) for x in (transition.value, transition.log_prob, transition.reward))
        self.counts[game] += 1
        self.episode_steps[game] += 1
        if transition.done:
            assert 1 <= transition.placement <= 8
            self.episodes[game] += 1
            self.episode_steps[game] = 0
            self.terminals += 1


def actor_stats(actor):
    sizes = actor.batch_sizes
    return {"decisions": sum(sizes), "forward_calls": len(sizes),
            "batch_mean": statistics.mean(sizes), "batch_median": statistics.median(sizes),
            "batch_p10": float(np.percentile(sizes, 10)), "batch_p90": float(np.percentile(sizes, 90)),
            "batch_max": max(sizes), "inference_seconds": actor.inference_seconds,
            "batch_sizes": sizes}


def synchronous(games, steps, learner, fixed, seed, threads, audit):
    env = MixedOpponentVectorEnv(games, seed=seed, threads=threads,
        opponent_policy=fixed, frozen_opponents=1, config={"turn_action_caps": [60]},
        reward_shaping={"operation_cost": 0.0001, "cycle_cost": 0.0002})
    obs = env.reset()
    torch.cuda.synchronize()
    started = time.monotonic()
    try:
        for step in range(steps):
            before = obs
            output = learner.act(before)
            obs, rewards, dones, infos = env.step(output.actions)
            for game in range(games):
                audit(ActorTransition(game, int(audit.episodes[game]), int(audit.episode_steps[game]), step,
                    {key: value[game] for key, value in before.items()}, int(output.actions[game]),
                    np.float32(output.values[game]), np.float32(output.log_probs[game]),
                    np.float32(rewards[game]), bool(dones[game]), int(infos[game].get("placement", 0)),
                    {key: value[game] for key, value in obs.items()}))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        return {"sampling_seconds": elapsed, "learner_samples": games * steps,
                "learner_samples_per_second": games * steps / elapsed,
                "native_steps": steps + env.waiting_rounds, "waiting_rounds": env.waiting_rounds,
                "terminal_transitions": audit.terminals}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    protected = {str(path.relative_to(ROOT)): sha(path) for path in (SOURCE, STOPPED)}
    assert json.loads(FORMAL.read_text())["status"] == "stopped_by_user"
    policies = {name: load_frozen_policy(SOURCE, "cuda", forward_precision="bf16")
                for name in ("learner", "bot_v1")}
    assert policies["learner"] is not policies["bot_v1"]
    versions = {name: [p._version for p in policy.parameters()] for name, policy in policies.items()}
    raw_dtypes = {name: set() for name in policies}
    hooks = []
    for name, policy in policies.items():
        assert not any(p.requires_grad for p in policy.parameters())
        hooks.append(policy.network.pointer[-1].register_forward_hook(
            lambda m, a, out, identity=name: raw_dtypes[identity].add(str(out.dtype))))
    warm = UnifiedActorSampler(args.games, PolicyActor(policies["learner"]), PolicyActor(policies["bot_v1"]),
                               seed=43, threads=args.threads, config={"turn_action_caps": [60]})
    for policy in policies.values():
        actor = PolicyActor(policy)
        for size in (2, 32, args.games):
            obs = {k: v[:size] for k, v in warm.learner_observation().items()}
            actor.act(obs)
    del warm
    rows = []
    sampler = None
    try:
        # ABBA order, two matched environment seeds; sampling order necessarily
        # changes RNG consumption and may change trajectories and action lengths.
        for mode, seed in (("synchronous", 43), ("unified", 43), ("unified", 44), ("synchronous", 44)):
            torch.manual_seed(1000 + seed)
            learner, fixed = (PolicyActor(policies[name]) for name in ("learner", "bot_v1"))
            audit = Audit(args.games)
            torch.cuda.reset_peak_memory_stats()
            write_json(FOLDER / "progress.json", {"status": "running", "mode": mode, "seed": seed,
                       "completed_cases": len(rows), "updated_at": now()})
            if mode == "synchronous":
                row = synchronous(args.games, args.steps, learner, fixed, seed, args.threads, audit)
            else:
                sampler = UnifiedActorSampler(args.games, learner, fixed, seed=seed, threads=args.threads,
                    config={"turn_action_caps": [60]}, reward_shaping={"operation_cost": .0001, "cycle_cost": .0002})
                torch.cuda.synchronize()
                row = sampler.collect(args.steps, audit)
                sampler = None
            assert (audit.counts == args.steps).all()
            assert sum(learner.batch_sizes) == args.games * args.steps
            row.update({"mode": mode, "seed": seed, "learner": actor_stats(learner),
                        "bot_v1": actor_stats(fixed), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(), "audit_passed": True})
            row["driver_seconds"] = row["sampling_seconds"] - learner.inference_seconds - fixed.inference_seconds
            rows.append(row)
            write_json(FOLDER / "partial-results.json", rows)
            print(json.dumps({"mode": mode, "seed": seed, "seconds": row["sampling_seconds"],
                  "samples_per_second": row["learner_samples_per_second"],
                  "bot_median_batch": row["bot_v1"]["batch_median"]}), flush=True)
        summary = {}
        for mode in ("synchronous", "unified"):
            selected = [row for row in rows if row["mode"] == mode]
            seconds = sum(row["sampling_seconds"] for row in selected)
            samples = sum(row["learner_samples"] for row in selected)
            summary[mode] = {"learner_samples": samples, "sampling_seconds": seconds,
                             "learner_samples_per_second": samples / seconds}
        speedup = summary["synchronous"]["sampling_seconds"] / summary["unified"]["sampling_seconds"]
        for name, policy in policies.items():
            assert versions[name] == [p._version for p in policy.parameters()]
            assert not policy.optimizer.state and all(p.grad is None for p in policy.parameters())
            assert raw_dtypes[name] == {"torch.bfloat16"}
        assert protected == {str(path.relative_to(ROOT)): sha(path) for path in (SOURCE, STOPPED)}
        assert json.loads(FORMAL.read_text())["status"] == "stopped_by_user"
        result = {"status": "completed", "completed_at": now(), "device": torch.cuda.get_device_name(),
            "games": args.games, "learner_steps_per_game_per_case": args.steps, "threads": args.threads,
            "lobby": {"scripted": 6, "fixed_v1": 1, "learner": 1}, "precision": "bf16 forward / fp32 outputs and weights",
            "raw_head_dtypes": {k: sorted(v) for k, v in raw_dtypes.items()},
            "optimizer_updates": 0, "models_saved": 0, "protected_model_hashes": protected,
            "formal_training_still_stopped": True, "summary": summary, "sampling_speedup": speedup,
            "sampling_time_reduction_fraction": 1 - 1 / speedup, "cases": rows,
            "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in
                (Path(__file__), ROOT / "python/taverngym/learning/actor_sampling.py",
                 ROOT / "python/taverngym/learning/vector_env.py")},
            "limitations": ["Standalone sampler; not integrated into PPO buffer/GAE/update loops.",
                "Same V1 source for both distinct policy identities; learner does not learn during this probe.",
                "Schedule and BF16 batch sizes change RNG consumption and trajectories; not paired identical games.",
                "Sampling only: no estimate of full-training speedup or learning quality.",
                "Rollout ends at the exact same per-game quota, including the final waiting tail."]}
        write_json(FOLDER / "results.json", result)
        write_json(FOLDER / "progress.json", {"status": "completed", "updated_at": now()})
        print(json.dumps({"summary": summary, "sampling_speedup": speedup}), flush=True)
    except Exception:
        write_json(FOLDER / "failure.json", {"at": now(), "traceback": traceback.format_exc(),
                   "native_failure": sampler.last_failure if sampler is not None else None, "completed_cases": rows})
        raise
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    main()
