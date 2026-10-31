"""Initialize and save the small entity PPO with zero training steps."""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from .env import make_vector_env
from .model import ModelConfig
from .policy import EntityPolicy
from .ppo import EntityPPO
from .precision import configure_precision
from .text import load_text_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo-small.json"))
    parser.add_argument("--text-cache", type=Path, default=Path("runs/assets/card-text/minilm-l6-v2.pt"))
    date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    parser.add_argument("--run-dir", type=Path, default=Path("runs") / f"{date}-ppo-small-init")
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir or Path("experiments") / args.run_dir.name
    checkpoint = args.run_dir / "initial_model.zip"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to replace initialized model: {checkpoint}")
    config = json.loads(args.config.read_text())
    model_config = ModelConfig(**config["model"])
    vectors, text_metadata = load_text_cache(args.text_cache)
    if vectors.shape[1] != model_config.text_dim:
        raise ValueError("Text cache dimension differs from the model configuration")
    if text_metadata["encoder"] != config["text_encoder"]:
        raise ValueError("Text cache encoder differs from the model configuration")
    if text_metadata["revision"] != config["text_encoder_revision"]:
        raise ValueError("Text cache revision differs from the model configuration")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    env = make_vector_env(config["n_envs"], config["seed"], reward_shaping=config.get("reward_shaping"))
    model = EntityPPO(
        EntityPolicy, env, **config["ppo"], seed=config["seed"], device=args.device,
        policy_kwargs={"model_config": asdict(model_config),
                       "card_data_sha256": text_metadata["card_data_sha256"],
                       "forward_precision": config.get("forward_precision", "fp32")},
        tensorboard_log=None, verbose=0,
    )
    model.policy.network.set_text_embeddings(vectors)
    configure_precision(model, config)
    model.rule_text_metadata = text_metadata
    model.experiment_config = config
    counts = model.policy.network.parameter_counts()
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert all(obs["action_mask"][i, int(a)] for i, a in enumerate(action))
    assert model.num_timesteps == 0 and not model.policy.optimizer.state
    args.run_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(exist_ok=True)
    model.save(checkpoint)
    # Load with no text-cache path: all three card representations live in the checkpoint.
    restored = EntityPPO.load(checkpoint, device=args.device)
    actual, _ = restored.predict(obs, deterministic=True)
    assert (action == actual).all()
    assert restored.num_timesteps == 0 and not restored.policy.optimizer.state
    assert torch.equal(model.policy.network.text_features, restored.policy.network.text_features)
    with torch.no_grad():
        tensors, _ = model.policy.obs_to_tensor(obs)
        before = model.policy.network(tensors)
        after = restored.policy.network(tensors)
        for expected, observed in zip(before, after):
            torch.testing.assert_close(expected, observed, rtol=1e-6, atol=1e-6)
    for name, expected in model.policy.state_dict().items():
        assert torch.equal(expected, restored.policy.state_dict()[name]), name
    resolved = dict(config, actual_parameters=counts["trainable"], rule_text=text_metadata,
                    initialization_device=args.device)
    (experiment_dir / "config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2) + "\n")
    result = {
        "status": "initialized", "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "checkpoint": str(checkpoint.resolve()), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_timesteps": model.num_timesteps, "optimizer_updates": len(model.policy.optimizer.state),
        "parameters": counts, "tokens": model.policy.network.layout.n_tokens,
        "text_cache": str(args.text_cache), "text_metadata": text_metadata,
        "device": str(model.device), "save_load_weights_exact_match": True,
        "save_load_outputs_match": True, "output_tolerance": {"rtol": 1e-6, "atol": 1e-6},
        "text_cache_required_for_loading": False,
    }
    (experiment_dir / "initialization.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
