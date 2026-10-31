"""Verify the actual saved zero-step model without changing its checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from taverngym.learning.env import make_vector_env
from taverngym.learning.ppo import EntityPPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", type=Path, default=Path("runs/20261031-ppo-small-init/initial_model.zip"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    before = hashlib.sha256(args.model.read_bytes()).hexdigest()
    # Loading the zip must not depend on the external rule-text cache or encoder.
    with patch("taverngym.learning.text.load_text_cache", side_effect=AssertionError("External text cache read")):
        model = EntityPPO.load(args.model, device=args.device)
    assert model.num_timesteps == 0 and not model.policy.optimizer.state
    env = make_vector_env(2, seed=1001)
    obs = env.reset()
    checked = 0
    for _ in range(16):
        actions, _ = model.predict(obs, deterministic=False)
        assert all(obs["action_mask"][i, int(a)] for i, a in enumerate(actions))
        obs, _, _, infos = env.step(actions)
        assert all(info.get("illegal") is None for info in infos)
        checked += len(actions)
    tensors, _ = model.policy.obs_to_tensor(obs)
    model.policy.set_training_mode(True)
    output = model.policy.network(tensors)
    valid_actions = output.logits.argmax(-1)
    # Synthetic loss targets exercise all heads; no optimizer step or training save.
    loss = (torch.nn.functional.cross_entropy(output.logits, valid_actions)
            + output.values.square().mean()
            + 0.1 * torch.nn.functional.cross_entropy(output.placement_logits, torch.tensor([0, 7], device=model.device)))
    loss.backward()
    modules = {"card_identity": model.policy.network.identity,
               "static_projection": model.policy.network.static_projection,
               "text_projection": model.policy.network.text_projection,
               "transformer": model.policy.network.transformer,
               "pointer": model.policy.network.pointer,
               "value": model.policy.network.value_head,
               "placement": model.policy.network.placement_head}
    for name, module in modules.items():
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters()), name
    assert all(torch.isfinite(p.grad).all() for p in model.policy.parameters() if p.grad is not None)
    assert model.policy.network.text_features.grad is None
    assert model.num_timesteps == 0 and not model.policy.optimizer.state
    after = hashlib.sha256(args.model.read_bytes()).hexdigest()
    assert before == after
    result = {
        "status": "passed", "device": str(model.device),
        "checkpoint_sha256": before, "checkpoint_unchanged": True,
        "training_timesteps": model.num_timesteps, "optimizer_updates": 0,
        "legal_native_actions_checked": checked,
        "loaded_without_external_text_cache": True,
        "all_heads_backward": "finite nonzero gradients",
        "backward_targets": "synthetic verification targets only; no optimizer step",
        "frozen_text_vectors": True,
    }
    Path(__file__).with_name(f"verification-{args.device}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
