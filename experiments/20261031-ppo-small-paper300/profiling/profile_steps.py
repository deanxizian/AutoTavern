"""Synchronized CPU/GPU latency breakdown matching the native PPO sampling path."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import obs_as_tensor

from taverngym.learning.buffer import CompactPlacementBuffer
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.vector_env import NativeVectorEnv


def summarize(values):
    data = np.array(values)
    return {"mean": float(data.mean()), "median": float(np.median(data)), "p95": float(np.quantile(data, .95))}


def main():
    folder = Path(__file__).resolve().parent
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    n_envs, steps = 256, 128
    env = NativeVectorEnv(n_envs, seed=700001, threads=4, config={"turn_action_caps": [60]})
    model = EntityPPO.load(
        "runs/20261031-ppo-small-init/initial_model.zip", env=env, device="cuda",
        n_steps=steps, batch_size=8192, n_epochs=1, micro_batch_size=256,
        rollout_buffer_class=CompactPlacementBuffer, rollout_buffer_kwargs={}, learning_rate=6e-6,
    )
    model.set_logger(configure(str(folder), ["stdout"]))
    model.policy.set_training_mode(False)
    obs = env.reset()
    episode_starts = np.ones(n_envs, dtype=bool)
    rows = []
    for index in range(16 + steps):
        start = time.perf_counter()
        masks = np.stack(env.env_method("action_masks"))
        after_masks = time.perf_counter()
        transfer_start, transfer_end, infer_start, infer_end = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        transfer_start.record()
        tensors = obs_as_tensor(obs, model.device)
        transfer_end.record()
        transfer_end.synchronize()
        after_transfer = time.perf_counter()
        infer_start.record()
        with torch.no_grad():
            actions, values, log_probs = model.policy(tensors, action_masks=masks)
        infer_end.record()
        infer_end.synchronize()
        after_infer = time.perf_counter()
        host_actions = actions.cpu().numpy()
        after_actions = time.perf_counter()
        next_obs, rewards, dones, infos = env.step(host_actions)
        after_env = time.perf_counter()
        if index >= 16:
            model.rollout_buffer.record_endings(dones, infos)
            model.rollout_buffer.add(obs, host_actions, rewards, episode_starts, values, log_probs,
                                     action_masks=masks)
        after_store = time.perf_counter()
        if index >= 16:
            rows.append({"mask_cpu_wall_ms": (after_masks - start) * 1000,
                         "h2d_wall_ms": (after_transfer - after_masks) * 1000,
                         "h2d_cuda_event_ms": transfer_start.elapsed_time(transfer_end),
                         "policy_wall_ms": (after_infer - after_transfer) * 1000,
                         "policy_cuda_event_ms": infer_start.elapsed_time(infer_end),
                         "action_d2h_wall_ms": (after_actions - after_infer) * 1000,
                         "env_cpu_wall_ms": (after_env - after_actions) * 1000,
                         "buffer_store_wall_ms": (after_store - after_env) * 1000,
                         "total_wall_ms": (after_store - start) * 1000})
        obs, episode_starts = next_obs, dones
    with torch.no_grad():
        final_values = model.policy.predict_values(obs_as_tensor(obs, model.device))
    gae_start = time.perf_counter()
    model.rollout_buffer.compute_returns_and_advantage(final_values, episode_starts)
    gae_ms = (time.perf_counter() - gae_start) * 1000
    model._current_progress_remaining = 1.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    update_start = time.perf_counter()
    model.train()
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - update_start
    summary = {key: summarize([row[key] for row in rows]) for key in rows[0]}
    result = {"n_envs": n_envs, "warmup_vector_steps": 16, "measured_vector_steps": steps,
              "measured_decisions": steps * n_envs, "seed": 700001,
              "units": "milliseconds per vector step (256 player decisions), unless stated otherwise",
              "sampling": summary,
              "amortized_per_decision_ms": summary["total_wall_ms"]["mean"] / n_envs,
              "gae_cpu_ms": gae_ms, "ppo_update_seconds": update_seconds,
              "effective_minibatch": 8192, "microbatch": 256, "accumulation": 32,
              "logical_optimizer_steps": steps * n_envs // 8192,
              "seconds_per_logical_optimizer_step": update_seconds / (steps * n_envs // 8192),
              "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
              "sampling_buffer_bytes": sum(array.nbytes for array in model.rollout_buffer._storage),
              "cuda_event_note": "Device event intervals include launch gaps, not a sum of kernel-only times; transfer and policy wall columns include synchronization.",
              "original_checkpoint_unchanged": True,
              "formal_training_budget_used": 0}
    (folder / "steps.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "steps-raw.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    model.rollout_buffer.close()
    env.close()


if __name__ == "__main__":
    main()
