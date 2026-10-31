"""Explicit entry point for training an initialized entity PPO checkpoint."""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from stable_baselines3.common.callbacks import CheckpointCallback

from .env import make_vector_env
from .ppo import DEFAULT_MICRO_BATCH_SIZE, EntityPPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--micro-batch-size", type=int, default=DEFAULT_MICRO_BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    args = parser.parse_args()
    experiment_dir = args.experiment_dir or Path("experiments") / args.run_dir.name
    if args.run_dir.exists():
        raise FileExistsError(f"Choose a new training run directory: {args.run_dir}")
    if args.total_timesteps < 1 or args.checkpoint_every < 1 or args.micro_batch_size < 1:
        raise ValueError("Training steps, checkpoint interval and microbatch size must be positive")
    torch.set_num_threads(2)
    model = EntityPPO.load(args.model, device=args.device, micro_batch_size=args.micro_batch_size)
    config = dict(model.experiment_config)
    if "ppo" in config:
        config["ppo"] = {**config["ppo"], "micro_batch_size": args.micro_batch_size}
    else:
        config["micro_batch_size"] = args.micro_batch_size
    model.experiment_config = config
    env = make_vector_env(args.n_envs if args.n_envs is not None else config["n_envs"], config["seed"])
    model.set_env(env)
    model.tensorboard_log = str(experiment_dir / "tensorboard")
    args.run_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir()
    resolved = dict(config, training_started=True, n_envs=env.num_envs,
                    source_checkpoint=str(args.model.resolve()),
                    source_checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    requested_additional_timesteps=args.total_timesteps)
    (experiment_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // env.num_envs),
        save_path=str(args.run_dir / "checkpoints"), name_prefix="ppo-small",
    )
    start = model.num_timesteps
    status = "completed"
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback,
                    reset_num_timesteps=False, tb_log_name="ppo-small")
    except KeyboardInterrupt:
        status = "interrupted"
    except BaseException:
        status = "failed"
        raise
    finally:
        name = "final_model.zip" if status == "completed" else f"{status}_model.zip"
        model.save(args.run_dir / name)
        result = {"status": status, "initial_timesteps": start,
                  "training_timesteps": model.num_timesteps, "model": name,
                  "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
        (experiment_dir / "training.json").write_text(json.dumps(result, indent=2) + "\n")
        env.close()


if __name__ == "__main__":
    main()
