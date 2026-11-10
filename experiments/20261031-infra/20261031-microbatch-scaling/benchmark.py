"""Isolated microbatch latency and memory measurements; preserves the training process."""

import argparse
import copy
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from taverngym.learning.buffer import CompactPlacementBuffer
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.vector_env import NativeVectorEnv


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def summary(values):
    array = np.array(values, dtype=np.float64)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p95": float(np.quantile(array, .95)), "samples": len(values)}


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


class Continue(BaseCallback):
    def _on_step(self):
        return True


class FixedBatch:
    """GPU-resident data, excluding decoding/H2D and holding the batch fixed across tests."""

    def __init__(self, batch):
        self.batch = batch
        self.values = batch.old_values.detach().cpu().numpy()
        self.returns = batch.returns.detach().cpu().numpy()

    def get(self, batch_size):
        assert batch_size == len(self.batch.actions)
        yield self.batch


def micro_forward_backward(model, batch, size):
    model.policy.optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start, middle, finish = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    wall_start = time.perf_counter()
    start.record()
    values, log_prob, entropy = model.policy.evaluate_actions(
        {key: value[:size] for key, value in batch.observations.items()},
        batch.actions[:size].long().flatten(), action_masks=batch.action_masks[:size],
    )
    middle.record()
    advantages = batch.advantages[:size]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ratio = (log_prob - batch.old_log_prob[:size]).exp()
    policy_loss = -torch.minimum(advantages * ratio, advantages * ratio.clamp(.8, 1.2)).mean()
    loss = policy_loss + .5 * torch.nn.functional.mse_loss(values.flatten(), batch.returns[:size]) - .002 * entropy.mean()
    known = batch.placement_targets[:size] >= 0
    placements = model.policy.placement_logits
    if known.any():
        loss = loss + .1 * torch.nn.functional.cross_entropy(placements[known], batch.placement_targets[:size][known])
    loss.backward()
    finish.record()
    finish.synchronize()
    elapsed = (time.perf_counter() - wall_start) * 1000
    result = {"forward_cuda_ms": start.elapsed_time(middle),
              "loss_and_backward_cuda_ms": middle.elapsed_time(finish),
              "forward_backward_cuda_ms": start.elapsed_time(finish),
              "forward_backward_wall_ms": elapsed}
    model.policy.placement_logits = None
    del values, log_prob, entropy, loss, placements
    return result


def benchmark(model, batch, size, policy_state, optimizer_state, warmup, repeats):
    model.policy.set_training_mode(True)
    model.micro_batch_size = size

    def restore():
        model.policy.optimizer.zero_grad(set_to_none=True)
        model.policy.load_state_dict(policy_state)
        model.policy.optimizer.load_state_dict(cpu_copy(optimizer_state))
        model._current_progress_remaining = 1.0
        torch.cuda.synchronize()

    restore()
    for _ in range(warmup):
        micro_forward_backward(model, batch, size)
    model.policy.optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    # Exclude allocator cold-start from the steady-state repetitions.
    micro_forward_backward(model, batch, size)
    model.policy.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    micro_rows = [micro_forward_backward(model, batch, size) for _ in range(repeats)]
    micro_peak = torch.cuda.max_memory_allocated()
    model.rollout_buffer = FixedBatch(batch)
    for _ in range(warmup):
        restore()
        model.train()
    model.policy.optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    restore()
    model.train()
    model.policy.optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    updates = []
    for _ in range(repeats):
        restore()
        start, finish = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start.record()
        begun = time.perf_counter()
        model.train()
        finish.record()
        finish.synchronize()
        updates.append({"wall_ms": (time.perf_counter() - begun) * 1000,
                        "cuda_interval_ms": start.elapsed_time(finish)})
    return {"status": "passed", "microbatch": size, "accumulation": 8192 // size,
            "micro": {key: summary([row[key] for row in micro_rows]) for key in micro_rows[0]},
            "effective_batch_update": {key: summary([row[key] for row in updates]) for key in updates[0]},
            "micro_peak_allocated_gib": micro_peak / 1024**3,
            "update_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "update_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "micro_raw": micro_rows, "update_raw": updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--model", type=Path, default=Path("runs/20261031-ppo-small-paper400/latest_model.zip"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    folder = args.output_dir or Path(__file__).resolve().parent
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "results.json").exists():
        raise FileExistsError("Choose a new output directory to retain previous measurements")
    if any(size < 2 or 8192 % size for size in args.sizes):
        raise ValueError("Microbatch sizes must divide 8192")
    process = Path(f"/proc/{args.training_pid}")
    if b"taverngym.learning.experiment" not in (process / "cmdline").read_bytes():
        raise RuntimeError("Training PID does not match the owned experiment")
    if "\nState:\tT" in (process / "status").read_text():
        raise RuntimeError("Training was already stopped; do not take over its pause")
    if os.getpgid(args.training_pid) != args.training_pid:
        raise RuntimeError("Training is not in its own process group")
    results = {"status": "running", "started_at": timestamp(), "training_pid": args.training_pid,
               "source_model": str(args.model), "source_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
               "microbatches": args.sizes, "effective_batch": 8192,
               "precision": "float32, matmul precision high, no AMP", "warmup": args.warmup,
               "repeats": args.repeats, "rows": [],
               "note": "CUDA event intervals include host launch gaps. Fixed GPU-resident batch excludes decoding/H2D; no formal model is modified."}
    paused = False
    paused_at = time.monotonic()
    original_buffer = None
    env = None

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        os.killpg(args.training_pid, signal.SIGSTOP)
        paused = True
        results["paused_at"] = timestamp()
        print("Training stopped temporarily; its in-memory state is preserved.", flush=True)
        time.sleep(.5)
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision("high")
        env = NativeVectorEnv(256, seed=800001, threads=4, config={"turn_action_caps": [60]})
        model = EntityPPO.load(args.model, env=env, device="cuda", n_steps=128,
                               batch_size=8192, n_epochs=1, micro_batch_size=256,
                               rollout_buffer_class=CompactPlacementBuffer, rollout_buffer_kwargs={})
        model.set_logger(configure(str(folder), []))
        _, callback = model._setup_learn(32768, Continue(), reset_num_timesteps=False)
        assert model.collect_rollouts(env, callback, model.rollout_buffer, 128)
        original_buffer = model.rollout_buffer
        np.random.seed(800001)
        batch = next(original_buffer.get(8192))
        results.update(device=torch.cuda.get_device_name(),
                       model_parameters=model.policy.network.parameter_counts()["trainable"],
                       collection_decisions=32768, seed=800001)
        policy_state = cpu_copy(model.policy.state_dict())
        optimizer_state = cpu_copy(model.policy.optimizer.state_dict())
        for size in results["microbatches"]:
            try:
                row = benchmark(model, batch, size, policy_state, optimizer_state, args.warmup, args.repeats)
            except torch.OutOfMemoryError as error:
                row = {"status": "out_of_memory", "microbatch": size, "error": str(error).splitlines()[0],
                       "note": "Paused training retains GPU memory; this is not a standalone capacity limit."}
                model.policy.optimizer.zero_grad(set_to_none=True)
                model.policy.placement_logits = None
                gc.collect()
                torch.cuda.empty_cache()
            results["rows"].append(row)
            (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({key: value for key, value in row.items() if not key.endswith("_raw")}), flush=True)
        assert hashlib.sha256(args.model.read_bytes()).hexdigest() == results["source_sha256"]
        results["status"] = "completed"
    finally:
        if original_buffer is not None:
            original_buffer.close()
        if env is not None:
            env.close()
        if paused:
            os.killpg(args.training_pid, signal.SIGCONT)
            results["training_resumed"] = True
            results["resumed_at"] = timestamp()
            results["pause_seconds"] = time.monotonic() - paused_at
            print("Formal training resumed.", flush=True)
            record = {"reason": "isolated microbatch scaling benchmark", "paused_at": results["paused_at"],
                      "resumed_at": results["resumed_at"], "seconds": results["pause_seconds"],
                      "record": str(folder / "results.json"),
                      "note": "This pause is included in the enclosing iteration's raw wall time."}
            with Path("experiments/20261031-infra/20261031-ppo-small-paper300/profiling-pauses.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        (folder / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
