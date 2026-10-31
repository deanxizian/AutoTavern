"""Run a recorded, iteration-budgeted PPO experiment from a saved initialization."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from .buffer import CompactPlacementBuffer
from .ppo import DEFAULT_MICRO_BATCH_SIZE, EntityPPO
from .schedule import LinearLearningRate
from .vector_env import NativeVectorEnv


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_model(model, path):
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.zip")
    model.save(temporary)
    temporary.replace(path)


def evaluate(model, games, seed, n_envs=32, threads=4, simulator_config=None):
    devices = [model.device.index or 0] if model.device.type == "cuda" else []
    # Evaluation must not advance the training sampler's random stream.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return _evaluate(model, games, seed, n_envs, threads, simulator_config)


def _evaluate(model, games, seed, n_envs, threads, simulator_config):
    if games == 0:
        return None
    env = NativeVectorEnv(min(n_envs, games), seed=seed, threads=threads, config=simulator_config)
    obs = env.reset()
    counts = np.zeros(env.num_envs, dtype=np.int64)
    quotas = np.array([(games + i) // env.num_envs for i in range(env.num_envs)])
    placements = []
    try:
        while (counts < quotas).any():
            actions, _ = model.predict(obs, deterministic=False)
            obs, _, dones, infos = env.step(actions)
            for i, done in enumerate(dones):
                if done and counts[i] < quotas[i]:
                    placements.append(infos[i]["placement"])
                    counts[i] += 1
    finally:
        env.close()
    values = np.array(placements)
    return {"games": games, "seed": seed, "deterministic": False,
            "mean_placement": float(values.mean()), "first_rate": float((values == 1).mean()),
            "top4_rate": float((values <= 4).mean()), "placements": placements,
            "opponents": "seven identical scripted bots"}


class Recorder(BaseCallback):
    def __init__(self, config, experiment, models):
        super().__init__()
        self.config, self.experiment, self.models = config, experiment, models
        self.started = time.monotonic()
        self.last_write = 0
        self.completed = 0
        self.best = float("inf")
        self.iteration_seconds = []
        self.phase = "sampling"

    def status(self, **extra):
        self.last_write = time.monotonic()
        elapsed = self.last_write - self.started
        budget = self.config["decisions_per_iteration"]
        estimate = (float(np.median(self.iteration_seconds)) * (self.config["iterations"] - self.completed)
                    if self.iteration_seconds else None)
        record = {"status": "running", "phase": self.phase, "updated_at": now(),
                  "pid": os.getpid(), "completed_iterations": self.completed,
                  "target_iterations": self.config["iterations"],
                  "sampled_decisions": self.model.num_timesteps,
                  "target_decisions": budget * self.config["iterations"],
                  "elapsed_seconds": elapsed, "estimated_remaining_seconds": estimate,
                  "latest_model": str(self.models / "latest_model.zip"), **extra}
        write_json(self.experiment / "progress.json", record)
        return record

    def _on_rollout_start(self):
        self.phase = "sampling"
        self.status()

    def _on_step(self):
        if time.monotonic() - self.last_write > 5:
            self.status()
        return True

    def _on_rollout_end(self):
        self.phase = "updating"
        self.status()

    def after_update(self, model):
        self.completed = model.num_timesteps // self.config["decisions_per_iteration"]
        seconds = model.last_rollout_seconds + model.last_train_seconds
        self.iteration_seconds.append(seconds)
        places = np.array(model._rollout_places)
        metrics = {key: float(value) for key, value in model.logger.name_to_value.items()
                   if isinstance(value, (float, int, np.number))}
        metrics.update(iteration=self.completed, decisions=model.num_timesteps,
                       rollout_seconds=model.last_rollout_seconds, update_seconds=model.last_train_seconds,
                       decisions_per_second=self.config["decisions_per_iteration"] / seconds,
                       games_finished=len(places))
        if len(places):
            metrics.update(mean_placement=float(places.mean()), first_rate=float((places == 1).mean()),
                           top4_rate=float((places <= 4).mean()))
        save_model(model, self.models / "latest_model.zip")
        every = self.config["checkpoint_every_iterations"]
        if self.completed % every == 0:
            save_model(model, self.models / "checkpoints" / f"iteration-{self.completed:04d}.zip")
        eval_every = self.config["evaluation_every_iterations"]
        if self.config["evaluation_games"] and self.completed % eval_every == 0:
            self.phase = "evaluating"
            self.status()
            result = evaluate(model, self.config["evaluation_games"], self.config["evaluation_seed"],
                              simulator_config=self.config.get("simulator_config"))
            write_json(self.experiment / f"evaluation-{self.completed:04d}.json", result)
            if result["mean_placement"] < self.best:
                self.best = result["mean_placement"]
                save_model(model, self.models / "best_model.zip")
            metrics["evaluation_mean_placement"] = result["mean_placement"]
        with (self.experiment / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        model.logger.dump(model.num_timesteps)
        self.phase = "iteration_complete"
        self.status(last_iteration=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume a fully updated iteration checkpoint")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.setdefault("micro_batch_size", DEFAULT_MICRO_BATCH_SIZE)
    expected = config["n_envs"] * config["n_steps"]
    if expected != config["decisions_per_iteration"] or expected % config["batch_size"]:
        raise ValueError("Iteration and minibatch budgets must divide exactly")
    if args.run_dir.exists() and not args.resume:
        raise FileExistsError(f"Choose a new model directory: {args.run_dir}")
    if args.resume and not args.run_dir.exists():
        raise FileNotFoundError("Resume requires an existing model directory")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    env = NativeVectorEnv(config["n_envs"], seed=config["seed"], threads=config["env_threads"],
                          config=config.get("simulator_config"))
    model = EntityPPO.load(
        args.model, env=env, device=args.device,
        learning_rate=LinearLearningRate(config["learning_rate_start"], config["learning_rate_end"]),
        n_steps=config["n_steps"], batch_size=config["batch_size"], n_epochs=config["n_epochs"],
        micro_batch_size=config["micro_batch_size"], rollout_buffer_class=CompactPlacementBuffer,
        rollout_buffer_kwargs={}, seed=config["seed"],
    )
    total_decisions = config["iterations"] * expected
    if not args.resume and model.num_timesteps != 0:
        raise ValueError("This experiment must start from the zero-step initialization")
    if model.num_timesteps % expected:
        raise ValueError("Resume requires a fully updated iteration checkpoint, not an interrupted rollout")
    remaining_decisions = total_decisions - model.num_timesteps
    if remaining_decisions <= 0:
        raise ValueError("Checkpoint has already reached the requested budget")
    initial_iteration = model.num_timesteps // expected
    model._current_progress_remaining = 1 - model.num_timesteps / total_decisions
    for group in model.policy.optimizer.param_groups:
        group["lr"] = model.lr_schedule(model._current_progress_remaining)
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "checkpoints").mkdir(parents=True, exist_ok=args.resume)
    model.experiment_config = config
    log_directory = args.experiment_dir
    previous_elapsed = 0
    if args.resume:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
        log_directory = args.experiment_dir / "sessions" / f"resume-{initial_iteration:04d}-{stamp}"
        previous_progress = args.experiment_dir / "progress.json"
        if previous_progress.exists():
            previous = json.loads(previous_progress.read_text())
            previous_elapsed = previous.get("elapsed_seconds", 0)
            previous_elapsed += max(0, (datetime.now(ZoneInfo("Asia/Shanghai"))
                                      - datetime.fromisoformat(previous["updated_at"])).total_seconds())
        history = args.experiment_dir / "resume-history"
        history.mkdir(exist_ok=True)
        write_json(history / f"{stamp}.json", {
            "resumed_at": now(), "source_model": str(args.model),
            "source_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "completed_iterations": initial_iteration, "target_iterations": config["iterations"],
            "remaining_decisions": remaining_decisions, "initial_optimizer_lr": model.lr_schedule(model._current_progress_remaining),
            "policy_and_adam_restored": True, "environment_and_rng_reinitialized": True,
            "log_directory": str(log_directory),
        })
    model.set_logger(configure(str(log_directory), ["stdout", "csv", "tensorboard"]))
    model.tensorboard_log = str(args.experiment_dir / "tensorboard")
    write_json(args.experiment_dir / "resolved-config.json", {
        **config, "source_model": str(args.model),
        "source_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "total_decisions": total_decisions, "resumed": args.resume,
        "initial_completed_iterations": initial_iteration,
        "remaining_decisions": remaining_decisions,
        "packed_observation_bytes": model.rollout_buffer.codec.bytes_per_observation(),
        "model_parameters": model.policy.network.parameter_counts()["trainable"],
        "ppo_inherited": {name: getattr(model, name) for name in
                          ("gamma", "gae_lambda", "ent_coef", "vf_coef", "aux_coef", "target_kl", "max_grad_norm")},
    })
    recorder = Recorder(config, args.experiment_dir, args.run_dir)
    recorder.init_callback(model)
    recorder.completed = initial_iteration
    recorder.started -= previous_elapsed
    if args.resume:
        metrics_path = args.experiment_dir / "metrics.jsonl"
        if metrics_path.exists():
            prior_metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            recorder.iteration_seconds = [row["rollout_seconds"] + row["update_seconds"] for row in prior_metrics
                                          if row["iteration"] <= initial_iteration]
        for evaluation_path in args.experiment_dir.glob("evaluation-*.json"):
            evaluation = json.loads(evaluation_path.read_text())
            recorder.best = min(recorder.best, evaluation.get("mean_placement", float("inf")))
    model._after_update = recorder.after_update
    if not args.resume:
        save_model(model, args.run_dir / "starting_model.zip")
    write_json(args.experiment_dir / "process.json", {"pid": os.getpid(), "started_at": now()})

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    status = "completed"
    try:
        if config["initial_evaluation_games"] and not args.resume:
            recorder.phase = "initial_evaluation"
            recorder.status()
            result = evaluate(model, config["initial_evaluation_games"], config["evaluation_seed"],
                              simulator_config=config.get("simulator_config"))
            recorder.best = result["mean_placement"]
            write_json(args.experiment_dir / "evaluation-0000.json", result)
            save_model(model, args.run_dir / "best_model.zip")
        model.learn(remaining_decisions, callback=recorder,
                    reset_num_timesteps=False, log_interval=None)
        save_model(model, args.run_dir / "final_model.zip")
        if config["final_evaluation_games"]:
            write_json(args.experiment_dir / "evaluation-final-heldout.json", evaluate(
                model, config["final_evaluation_games"], config["final_evaluation_seed"],
                simulator_config=config.get("simulator_config")))
    except KeyboardInterrupt:
        status = "interrupted"
        save_model(model, args.run_dir / "interrupted_model.zip")
    except BaseException:
        status = "failed"
        save_model(model, args.run_dir / "failed_model.zip")
        raise
    finally:
        result = recorder.status(status=status,
                                 peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
        write_json(args.experiment_dir / "training.json", result)
        model.rollout_buffer.close()
        env.close()


if __name__ == "__main__":
    main()
