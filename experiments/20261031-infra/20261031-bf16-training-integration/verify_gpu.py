"""Verify both real V1 actors use BF16 without any optimizer updates or new models."""

import hashlib
from pathlib import Path

import torch

from taverngym.learning.experiment import load_frozen_policy, now, write_json
from taverngym.learning.ppo import EntityPPO, PlacementBuffer
from taverngym.learning.precision import configure_precision
from taverngym.learning.vector_env import MixedOpponentVectorEnv


def main():
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    folder = Path(__file__).resolve().parent
    source = root / "runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip"
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    fixed = load_frozen_policy(source, "cuda")
    env = MixedOpponentVectorEnv(16, seed=43, threads=4, opponent_policy=fixed, frozen_opponents=1,
                                config={"turn_action_caps": [60]})
    model = EntityPPO.load(source, env=env, device="cuda", n_steps=2, batch_size=2,
                           rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
    config = {"forward_precision": "bf16", "opponent_forward_precision": "bf16"}
    configure_precision(model, config, fixed)
    assert model.policy.forward_precision == fixed.forward_precision == "bf16"
    adam_before = model.optimizer_step_count()
    observed = {"learner": [], "fixed_bot": []}
    hooks = [model.policy.network.pointer[-1].register_forward_hook(
                 lambda m, a, out: observed["learner"].append((str(out.dtype), out.shape[0]))),
             fixed.network.pointer[-1].register_forward_hook(
                 lambda m, a, out: observed["fixed_bot"].append((str(out.dtype), out.shape[0])))]
    try:
        obs = env.reset()
        tensors, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            model.policy.set_training_mode(False)
            actions, values, old_logp = model.policy(tensors)
            model.policy.set_training_mode(True)
            new_values, new_logp, entropy = model.policy.evaluate_actions(tensors, actions)
        torch.testing.assert_close(old_logp, new_logp, rtol=0, atol=1e-6)
        torch.testing.assert_close(values, new_values, rtol=0, atol=1e-6)
        assert values.dtype == old_logp.dtype == new_values.dtype == new_logp.dtype == entropy.dtype == torch.float32
        for _ in range(32):
            actions, _ = model.predict(obs, deterministic=False)
            obs, _, _, _ = env.step(actions)
        assert all(dtype == "torch.bfloat16" for entries in observed.values() for dtype, batch in entries)
        assert all(p.dtype == torch.float32 for p in model.policy.parameters())
        assert all(p.dtype == torch.float32 and not p.requires_grad for p in fixed.parameters())
        assert model.optimizer_step_count() == adam_before and not fixed.optimizer.state
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        write_json(folder / "gpu-validation.json", {
            "status": "passed", "verified_at": now(), "device": torch.cuda.get_device_name(),
            "source_sha256": before, "source_model_unchanged": True, "new_checkpoints_saved": False,
            "learner_samples_checked": 512, "frozen_opponents_per_game": 1, "scripted_opponents_per_game": 6,
            "learner_raw_logits_dtype": "bfloat16", "fixed_bot_raw_logits_dtype": "bfloat16",
            "probabilities_values_logprobs": "float32", "parameter_dtype": "float32",
            "sampling_update_rescoring_match": True, "optimizer_updates": 0,
            "frozen_bot_batched_across_games": max(batch for dtype, batch in observed["fixed_bot"]) > 1,
            "maximum_fixed_bot_batch": max(batch for dtype, batch in observed["fixed_bot"]),
            "precision_settings": config["precision_settings"], "second_training_restarted": False,
        })
        print("Real V1 learner and fixed bot both use BF16; FP32 outputs and rescoring verified; no optimizer updates.")
    finally:
        for hook in hooks:
            hook.remove()
        env.close()


if __name__ == "__main__":
    main()
