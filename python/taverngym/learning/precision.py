"""Precision settings shared by rollout, PPO updates and fixed opponents."""

import torch

FORWARD_PRECISIONS = ("fp32", "bf16")


def validate_forward_precision(value):
    if value not in FORWARD_PRECISIONS:
        raise ValueError("forward_precision must be fp32 or bf16")
    return value


def configure_precision(model, config, opponent_policy=None):
    """Resolve explicit settings or inherit a checkpoint, and record actual modes.

    Legacy policies default to FP32. Choosing BF16 for a new stage also applies
    to fixed opponents unless their precision is explicitly set independently.
    Only operator precision changes; parameters/gradients/Adam remain FP32.
    """
    model.set_forward_precision(config.get("forward_precision", model.policy.forward_precision))
    config["forward_precision"] = model.policy.forward_precision
    if opponent_policy is not None and hasattr(opponent_policy, "set_forward_precision"):
        opponent_policy.set_forward_precision(config.get("opponent_forward_precision", model.policy.forward_precision))
        config["opponent_forward_precision"] = opponent_policy.forward_precision
    config["precision_settings"] = {
        "learner_sampling_forward": model.policy.forward_precision,
        "learner_update_forward": model.policy.forward_precision,
        "opponent_forward": config.get("opponent_forward_precision"),
        "parameters": "fp32", "gradient_accumulation": "fp32", "adam_state": "fp32",
        "head_outputs_and_probabilities": "fp32", "gae_returns_losses": "fp32",
        "checkpoint_floating_weights": "fp32", "grad_scaler": False,
        "fp32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    config["precision"] = (
        f"{model.policy.forward_precision.upper()} neural forward; FP32 head outputs, probabilities, "
        "GAE, losses, parameters, gradient accumulation and Adam; no GradScaler"
    )
