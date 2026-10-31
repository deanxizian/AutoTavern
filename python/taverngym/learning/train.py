"""Explicit entry point for training an initialized entity PPO checkpoint."""

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from .env import make_vector_env
from .ppo import DEFAULT_MICRO_BATCH_SIZE, EntityPPO
from .precision import FORWARD_PRECISIONS, configure_precision
from .rewards import RewardShapingConfig
from .tracking import WandbOutputFormat, WandbTracker, tracking_settings


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
    parser.add_argument("--tracking-backend", choices=("wandb", "none"), default="wandb")
    parser.add_argument("--forward-precision", choices=FORWARD_PRECISIONS,
                        help="Neural forward precision; missing inherits checkpoint")
    parser.add_argument("--operation-cost", type=float, help="Move/Freeze reward cost; enables reward shaping")
    parser.add_argument("--cycle-cost", type=float, help="Additional no-progress return cost; enables reward shaping")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir or Path("experiments") / args.run_dir.name
    if args.run_dir.exists():
        raise FileExistsError(f"Choose a new training run directory: {args.run_dir}")
    if args.total_timesteps < 1 or args.checkpoint_every < 1 or args.micro_batch_size < 1:
        raise ValueError("Training steps, checkpoint interval and microbatch size must be positive")
    torch.set_num_threads(2)
    model = EntityPPO.load(args.model, device=args.device, micro_batch_size=args.micro_batch_size)
    config = dict(model.experiment_config)
    if args.forward_precision is not None:
        config["forward_precision"] = args.forward_precision
    configure_precision(model, config)
    if "ppo" in config:
        config["ppo"] = {**config["ppo"], "micro_batch_size": args.micro_batch_size}
    else:
        config["micro_batch_size"] = args.micro_batch_size
    if args.operation_cost is not None or args.cycle_cost is not None:
        costs = dict(config.get("reward_shaping") or {})
        if args.operation_cost is not None:
            costs["operation_cost"] = args.operation_cost
        if args.cycle_cost is not None:
            costs["cycle_cost"] = args.cycle_cost
        config["reward_shaping"] = asdict(RewardShapingConfig(**costs))
    elif config.get("reward_shaping") is not None:
        config["reward_shaping"] = asdict(RewardShapingConfig(**config["reward_shaping"]))
    model.experiment_config = config
    env = make_vector_env(args.n_envs if args.n_envs is not None else config["n_envs"], config["seed"],
                          reward_shaping=config.get("reward_shaping"))
    model.set_env(env)
    model.tensorboard_log = None
    args.run_dir.mkdir(parents=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir()
    resolved = dict(config, training_started=True, n_envs=env.num_envs,
                    tracking_backend=args.tracking_backend, tracking=tracking_settings(config),
                    source_checkpoint=str(args.model.resolve()),
                    source_checkpoint_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    requested_additional_timesteps=args.total_timesteps)
    (experiment_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    tracker = WandbTracker(experiment_dir, resolved) if args.tracking_backend == "wandb" else None
    logger = configure(str(experiment_dir), ["stdout", "csv"])
    if tracker is not None:
        logger.output_formats.append(WandbOutputFormat(tracker))
    model.set_logger(logger)

    def record_optimizer_step(current, metrics, step):
        if tracker is not None:
            tracker.log({**{f"train/{name}": value for name, value in metrics.items()},
                         "train/learning_rate": current.policy.optimizer.param_groups[0]["lr"],
                         "train/sampled_decisions": current.num_timesteps}, step)

    def record_iteration(current):
        for name, value in list(logger.name_to_value.items()):
            if name.startswith("train/"):
                if name != "train/n_updates":
                    logger.record(f"ppo_summary/{name.removeprefix('train/')}", value)
                logger.record(name, value, exclude="wandb")
        places = current._rollout_places
        if places:
            logger.record("rollout/mean_placement", sum(places) / len(places))
            logger.record("rollout/top4_rate", sum(p <= 4 for p in places) / len(places))
            logger.record("rollout/first_rate", sum(p == 1 for p in places) / len(places))
        logger.record("time/rollout_seconds", current.last_rollout_seconds)
        logger.record("time/update_seconds", current.last_train_seconds)
        logger.dump(current.optimizer_step_count())

    model._after_optimizer_step = record_optimizer_step
    model._after_update = record_iteration
    callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // env.num_envs),
        save_path=str(args.run_dir / "checkpoints"), name_prefix="ppo-small",
    )
    start = model.num_timesteps
    status = "completed"
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback,
                    reset_num_timesteps=False, log_interval=None)
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
        if tracker is not None:
            tracker.finish(exit_code=0 if status == "completed" else 1)


if __name__ == "__main__":
    main()
