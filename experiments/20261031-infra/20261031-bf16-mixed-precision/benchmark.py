"""Bounded FP32 versus BF16 autocast comparison. Formal training stays stopped."""

import argparse
from datetime import datetime
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from taverngym.learning.buffer import CompactPlacementBuffer
from taverngym.learning.model import ModelOutput
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.vector_env import NativeVectorEnv


SOURCE = Path("runs/20261031-ppo-small-paper300/checkpoints/iteration-0003-before-budget-change.zip")
FOLDER = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "microbatch_helpers", "experiments/20261031-infra/20261031-microbatch-scaling/benchmark.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class Precision:
    def __init__(self, network):
        self.original = network.forward
        self.mode = "fp32"
        self.observed = {}
        network.forward = self.forward

    def forward(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.mode == "bf16"):
            output = self.original(*args, **kwargs)
        # Probability normalization, log-probabilities, ratios, KL and losses stay FP32.
        self.observed = {"raw_policy_logits": str(output.logits.dtype),
                         "raw_values": str(output.values.dtype),
                         "raw_placement_logits": str(output.placement_logits.dtype)}
        return ModelOutput(output.logits.float(), output.values.float(), output.placement_logits.float())


def inference(model, obs, precision, repeats=15):
    model.policy.set_training_mode(False)
    subset = {key: value[:256] for key, value in obs.items()}
    for _ in range(4):
        with torch.no_grad():
            model.policy(subset)
    torch.cuda.synchronize()
    rows = []
    for _ in range(repeats):
        start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        begun = time.perf_counter()
        start.record()
        with torch.no_grad():
            model.policy(subset)
        end.record()
        end.synchronize()
        rows.append({"cuda_ms": start.elapsed_time(end), "wall_ms": (time.perf_counter() - begun) * 1000})
    return {"batch": 256, "timing": {key: helpers.summary([row[key] for row in rows]) for key in rows[0]},
            "raw": rows, "raw_network_dtypes": precision.observed.copy(),
            "distribution_dtype": str(model.policy.action_dist.distribution.logits.dtype)}


def capture(model, batch, training=False):
    model.policy.set_training_mode(training)
    outputs = {key: [] for key in ("log_prob", "values", "placement", "logits")}
    for begin in range(0, len(batch.actions), 256):
        span = slice(begin, begin + 256)
        obs = {key: value[span] for key, value in batch.observations.items()}
        with torch.set_grad_enabled(training):
            values, log_prob, _ = model.policy.evaluate_actions(obs, batch.actions[span].long().flatten(),
                                                              action_masks=batch.action_masks[span])
            current = {"log_prob": log_prob, "values": values.flatten(),
                       "placement": model.policy.placement_logits,
                       "logits": model.policy.action_dist.distribution.logits}
            for key, value in current.items():
                outputs[key].append(value.detach().cpu())
        model.policy.placement_logits = None
        model.policy.action_dist.distribution = None
    return {key: torch.cat(value) for key, value in outputs.items()}


def gradient(model, batch):
    model.policy.set_training_mode(True)
    model.policy.optimizer.zero_grad(set_to_none=True)
    size = 256
    values, log_prob, entropy = model.policy.evaluate_actions(
        {key: value[:size] for key, value in batch.observations.items()},
        batch.actions[:size].long().flatten(), action_masks=batch.action_masks[:size])
    advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    ratio = (log_prob - batch.old_log_prob[:size]).exp()
    policy_loss = -torch.minimum(advantages[:size] * ratio, advantages[:size] * ratio.clamp(.8, 1.2)).mean()
    loss = policy_loss + .5 * torch.nn.functional.mse_loss(values.flatten(), batch.returns[:size]) - .002 * entropy.mean()
    known = batch.placement_targets[:size] >= 0
    placement = model.policy.placement_logits
    if known.any():
        loss = loss + .1 * torch.nn.functional.cross_entropy(placement[known], batch.placement_targets[:size][known])
    loss.backward()
    result = torch.cat([parameter.grad.detach().flatten().cpu() if parameter.grad is not None else
                        torch.zeros(parameter.numel()) for parameter in model.policy.parameters()])
    assert torch.isfinite(result).all()
    model.policy.placement_logits = None
    return result, float(loss.detach().item())


def numerical_check(model, batch, precision, policy_state):
    precision.mode = "fp32"
    model.policy.load_state_dict(policy_state)
    fp_eval = capture(model, batch)
    fp_gradient, fp_loss = gradient(model, batch)
    precision.mode = "bf16"
    model.policy.load_state_dict(policy_state)
    bf_eval = capture(model, batch)
    bf_training = capture(model, batch, training=True)
    bf_gradient, bf_loss = gradient(model, batch)
    difference = bf_eval["log_prob"] - fp_eval["log_prob"]
    probability = fp_eval["logits"].exp()
    kl = (probability * (fp_eval["logits"] - bf_eval["logits"])).sum(-1)
    self_ratio_log = bf_training["log_prob"] - bf_eval["log_prob"]
    cross_ratio_log = bf_training["log_prob"] - batch.old_log_prob.cpu()
    delta = bf_gradient - fp_gradient
    return {
        "samples": len(batch.actions), "gradient_samples": 256,
        "fp32_loss": fp_loss, "bf16_loss": bf_loss,
        "log_prob_abs_difference_mean": float(difference.abs().mean()),
        "log_prob_abs_difference_max": float(difference.abs().max()),
        "log_prob_abs_difference_p95": float(torch.quantile(difference.abs(), .95)),
        "exact_policy_kl_fp32_to_bf16_mean": float(kl.mean()),
        "exact_policy_kl_fp32_to_bf16_max": float(kl.max()),
        "greedy_action_change_rate": float((fp_eval["logits"].argmax(-1) != bf_eval["logits"].argmax(-1)).float().mean()),
        "value_abs_difference_mean": float((fp_eval["values"] - bf_eval["values"]).abs().mean()),
        "value_abs_difference_max": float((fp_eval["values"] - bf_eval["values"]).abs().max()),
        "same_bf16_eval_train_log_prob_abs_max": float(self_ratio_log.abs().max()),
        "same_bf16_eval_train_approx_kl": float((self_ratio_log.exp() - 1 - self_ratio_log).mean()),
        "fp32_rollout_bf16_train_approx_kl": float((cross_ratio_log.exp() - 1 - cross_ratio_log).mean()),
        "gradient_finite": True, "gradient_relative_l2_difference": float(delta.norm() / fp_gradient.norm()),
        "gradient_cosine_similarity": float(torch.nn.functional.cosine_similarity(fp_gradient, bf_gradient, dim=0)),
        "parameter_dtypes": sorted(set(str(p.dtype) for p in model.policy.parameters())),
        "gradient_dtypes": sorted(set(str(p.grad.dtype) for p in model.policy.parameters() if p.grad is not None)),
        "optimizer_moment_dtypes": sorted(set(str(state[key].dtype) for state in model.policy.optimizer.state.values()
                                               for key in ("exp_avg", "exp_avg_sq"))),
        "note": "Finite gradients and small one-batch error do not establish long-horizon training equivalence.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FOLDER)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--modes", nargs="+", choices=["fp32", "bf16"], default=["fp32", "bf16"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    if any(size < 2 or 8192 % size for size in args.sizes):
        raise ValueError("Microbatch sizes must divide 8192")
    if (folder / "results.json").exists():
        raise FileExistsError("Results already exist; retain the previous measurement")
    assert torch.cuda.is_bf16_supported(), "CUDA device must support BF16"
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
              "source_model": str(SOURCE), "source_sha256": source_hash,
              "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "master_parameters": "FP32", "gradients": "FP32", "adam_moments": "FP32",
              "bf16_scope": "autocast neural forward only; outputs, distributions and PPO losses are FP32",
              "effective_batch": 8192, "warmup": args.warmup, "repeats": args.repeats,
              "microbatches": args.sizes, "mode_order": args.modes, "rows": [],
              "formal_training_started": False, "saved_test_weights": False,
              "timing_note": "GPU-resident batch; excludes sampling, decoding and H2D. CUDA event spans include launch gaps."}
    env = NativeVectorEnv(256, seed=900101, threads=4, config={"turn_action_caps": [60]})
    model = EntityPPO.load(SOURCE, env=env, device="cuda", n_steps=128, batch_size=8192,
                           n_epochs=1, micro_batch_size=256, rollout_buffer_class=CompactPlacementBuffer,
                           rollout_buffer_kwargs={})
    model.set_logger(configure(str(folder), []))
    _, callback = model._setup_learn(32768, helpers.Continue(), reset_num_timesteps=False)
    assert model.collect_rollouts(env, callback, model.rollout_buffer, 128)
    original_buffer = model.rollout_buffer
    np.random.seed(900101)
    batch = next(original_buffer.get(8192))
    policy_state = helpers.cpu_copy(model.policy.state_dict())
    optimizer_state = helpers.cpu_copy(model.policy.optimizer.state_dict())
    precision = Precision(model.policy.network)
    result["sampling_inference"] = {}
    for mode in ("fp32", "bf16"):
        precision.mode = mode
        result["sampling_inference"][mode] = inference(model, batch.observations, precision)
    print("Inference comparison completed.", flush=True)
    result["numerical"] = numerical_check(model, batch, precision, policy_state)
    print("Numerical checks:", json.dumps(result["numerical"]), flush=True)
    (folder / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for size in args.sizes:
        for mode in args.modes:
            precision.mode = mode
            try:
                row = helpers.benchmark(model, batch, size, policy_state, optimizer_state,
                                        args.warmup, args.repeats)
                row["precision"] = mode
                row["network_output_dtypes"] = precision.observed.copy()
            except torch.OutOfMemoryError as error:
                row = {"status": "out_of_memory", "microbatch": size, "precision": mode,
                       "error": str(error).splitlines()[0]}
                model.policy.optimizer.zero_grad(set_to_none=True)
                model.policy.placement_logits = None
                gc.collect()
                torch.cuda.empty_cache()
            result["rows"].append(row)
            (folder / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({key: value for key, value in row.items() if not key.endswith("_raw")}), flush=True)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    result["status"] = "completed"
    result["completed_at"] = datetime.now().astimezone().isoformat()
    result["source_unchanged"] = True
    original_buffer.close()
    env.close()
    (folder / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Mixed-precision benchmark completed; formal training remains stopped.", flush=True)


if __name__ == "__main__":
    main()
