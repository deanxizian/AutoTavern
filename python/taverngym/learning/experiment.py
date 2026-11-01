"""Run a recorded, iteration-budgeted PPO experiment from a saved initialization."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime
from dataclasses import asdict
from zoneinfo import ZoneInfo

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

from .buffer import CompactPlacementBuffer
from .ppo import DEFAULT_MICRO_BATCH_SIZE, EntityPPO, PlacementBuffer
from .precision import FORWARD_PRECISIONS, configure_precision
from .rewards import REWARD_METRICS, RewardShapingConfig, reward_protocols_equal
from .schedule import learning_rate_schedule
from .opponents import FrozenOpponentPool, OpponentGroup, opponent_groups
from .tracking import WandbOutputFormat, WandbTracker, tracking_settings
from .vector_env import MixedOpponentVectorEnv, NativeVectorEnv
from .actor_sampling import ActorOpponentVectorEnv

EVALUATION_METRICS = (*REWARD_METRICS, "cap_end_rate")


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


def evaluate(model, games, seed, n_envs=32, threads=4, simulator_config=None, reward_shaping=None,
             opponent_policy=None, deterministic=False, frozen_opponents=4, rotate_seats=False):
    devices = [model.device.index or 0] if model.device.type == "cuda" else []
    # Evaluation must not advance the training sampler's random stream.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return _evaluate(model, games, seed, n_envs, threads, simulator_config, reward_shaping,
                         opponent_policy, deterministic, frozen_opponents, rotate_seats)


def _evaluate(model, games, seed, n_envs, threads, simulator_config, reward_shaping,
              opponent_policy, deterministic, frozen_opponents, rotate_seats):
    if games == 0:
        return None
    options = dict(seed=seed, threads=threads, config=simulator_config,
                   reward_shaping=reward_shaping, rotate_seats=rotate_seats)
    env = (MixedOpponentVectorEnv(min(n_envs, games), opponent_policy=opponent_policy,
                                  frozen_opponents=frozen_opponents, **options)
           if opponent_policy is not None else NativeVectorEnv(min(n_envs, games), **options))
    obs = env.reset()
    counts = np.zeros(env.num_envs, dtype=np.int64)
    quotas = np.array([(games + i) // env.num_envs for i in range(env.num_envs)])
    placements = []
    components = []
    opponent_terminations = []
    episode_records = []
    try:
        while (counts < quotas).any():
            actions, _ = model.predict(obs, deterministic=deterministic)
            turns_before_action = np.rint(obs["features"][:, 0] * 20).astype(np.int64)
            obs, _, dones, infos = env.step(actions)
            for i, done in enumerate(dones):
                if done and counts[i] < quotas[i]:
                    placements.append(infos[i]["placement"])
                    if "reward_components" in infos[i]:
                        components.append(infos[i]["reward_components"])
                    opponent_terminations.append(infos[i].get("opponent_terminations", []))
                    episode_records.append({"placement": int(infos[i]["placement"]),
                        "turn_before_terminal_action": int(turns_before_action[i]),
                        "episode": {key: infos[i]["episode"][key] for key in ("r", "l")},
                        "termination_reason": infos[i].get("termination_reason", "native_terminal"),
                        "reward_components": infos[i].get("reward_components"),
                        "learner_seat": infos[i].get("learner_seat")})
                    counts[i] += 1
    finally:
        env.close()
    values = np.array(placements)
    result = {"games": games, "seed": seed, "deterministic": deterministic,
              "mean_placement": float(values.mean()), "first_rate": float((values == 1).mean()),
              "top4_rate": float((values <= 4).mean()), "placements": placements,
              "opponents": "seven identical scripted bots"}
    result["episodes"] = episode_records
    if opponent_policy is not None:
        result["opponents"] = f"{frozen_opponents} frozen policies and {7 - frozen_opponents} scripted bots"
        if isinstance(opponent_policy, FrozenOpponentPool):
            result["opponent_groups"] = [{"name": g.name, "count": g.count, "deterministic": False,
                                          "reward_shaping": g.reward_shaping}
                                         for g in opponent_policy.groups]
            result["opponents"] = " + ".join(
                [*(f"{g.count} fixed {g.name}" for g in opponent_policy.groups),
                 f"{7 - frozen_opponents} scripted bots"])
    if components:
        result.update({name: float(np.mean([row[name] for row in components]))
                       for name in EVALUATION_METRICS})
        result["reward_shaping"] = asdict(RewardShapingConfig(**reward_shaping))
    if any(opponent_terminations):
        result["opponent_terminations"] = opponent_terminations
    return result


def load_frozen_policy(path, device, forward_precision=None, *, reward_shaping=None,
                       model_version=None):
    # A small temporary buffer avoids allocating the old training rollout.
    frozen = EntityPPO.load(path, device=device, n_envs=1, n_steps=2, batch_size=2,
                            rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
    if reward_shaping is not None:
        protocol = getattr(frozen, "experiment_config", {})
        if (not reward_protocols_equal(protocol.get("reward_shaping"), reward_shaping)
                or protocol.get("model_version") != model_version):
            raise ValueError("Frozen opponent termination protocol differs from its checkpoint")
    if forward_precision is not None:
        frozen.set_forward_precision(forward_precision)
    policy = frozen.policy
    policy.optimizer.state.clear()
    policy.requires_grad_(False)
    policy.set_training_mode(False)
    return policy


def load_frozen_opponents(config, device):
    """Load each version once; all seats of that version share its frozen policy."""
    if config.get("opponent_pool"):
        groups = []
        for spec in config["opponent_pool"]:
            path = Path(spec["model"])
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if source_hash != spec["sha256"]:
                raise ValueError(f"Frozen opponent checkpoint hash changed: {spec['name']}")
            groups.append(OpponentGroup(spec["name"], load_frozen_policy(path, device,
                                        reward_shaping=spec.get("reward_shaping"),
                                        model_version=spec.get("model_version")), spec["count"],
                                        spec.get("reward_shaping")))
        pool = FrozenOpponentPool(tuple(groups))
        opponent_groups(pool, config["frozen_opponents"])
        return pool
    if config.get("opponent_model"):
        path = Path(config["opponent_model"])
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if config.get("opponent_model_sha256", source_hash) != source_hash:
            raise ValueError("Frozen opponent checkpoint hash changed")
        config["opponent_model_sha256"] = source_hash
        return load_frozen_policy(path, device)
    return None


def evaluate_suite(model, games, seed, config, opponent_policy=None):
    # Standalone held-out evaluation must use the same configured precision as
    # training validation, even when opponents originate from an old FP32 file.
    configure_precision(model, config, opponent_policy)
    options = dict(simulator_config=config.get("simulator_config"),
                   reward_shaping=config.get("reward_shaping"))
    if not config.get("evaluation_matrix"):
        return evaluate(model, games, seed, **options)
    if opponent_policy is None:
        raise ValueError("The mixed evaluation matrix requires the frozen opponent")
    cases = [("mixed_sampled", opponent_policy, False),
             ("mixed_greedy", opponent_policy, True)]
    # Historical configs keep their four-condition protocol. New templates
    # explicitly disable the additional seven-scripted-opponent evaluations.
    if config.get("evaluation_include_scripted", True):
        cases.extend((("scripted_sampled", None, False), ("scripted_greedy", None, True)))
    conditions = {}
    for name, opponent, greedy in cases:
        conditions[name] = evaluate(
            model, games, seed, opponent_policy=opponent, deterministic=greedy,
            frozen_opponents=config.get("frozen_opponents", 4), rotate_seats=True, **options,
        )
    return {**conditions["mixed_sampled"], "conditions": conditions,
            "selection_condition": "mixed_sampled"}


def log_evaluation(logger, prefix, result):
    names = ("mean_placement", "first_rate", "top4_rate", *EVALUATION_METRICS)
    conditions = result.get("conditions", {"mixed_sampled": result})
    labels = {"mixed_sampled": "", "mixed_greedy": "greedy_",
              "scripted_sampled": "scripted_", "scripted_greedy": "scripted_greedy_"}
    for label, row in conditions.items():
        for name in names:
            if name in row:
                logger.record(f"{prefix}/{labels[label]}{name}", row[name])


class Recorder(BaseCallback):
    def __init__(self, config, experiment, models, opponent_policy=None):
        super().__init__()
        self.config, self.experiment, self.models = config, experiment, models
        self.opponent_policy = opponent_policy
        self.started = time.monotonic()
        self.last_write = 0
        self.completed = 0
        self.best = float("inf")
        self.best_model_path = models / config.get("best_model_filename", "best_model.zip")
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
                  "optimizer_steps": self.model.optimizer_step_count(),
                  "target_decisions": budget * self.config["iterations"],
                  "elapsed_seconds": elapsed, "estimated_remaining_seconds": estimate,
                  "latest_model": str(self.models / "latest_model.zip"), **extra}
        record["best_model"] = str(self.best_model_path)
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
        model.logger.record("progress/ppo_iterations", self.completed)
        model.logger.record("progress/sampled_decisions", model.num_timesteps)
        model.logger.record("time/rollout_seconds", model.last_rollout_seconds)
        model.logger.record("time/update_seconds", model.last_train_seconds)
        if len(places):
            for name in ("mean_placement", "first_rate", "top4_rate"):
                model.logger.record(f"rollout/{name}", metrics[name])
        save_model(model, self.models / "latest_model.zip")
        every = self.config["checkpoint_every_iterations"]
        if self.completed % every == 0:
            save_model(model, self.models / "checkpoints" / f"iteration-{self.completed:04d}.zip")
        eval_every = self.config["evaluation_every_iterations"]
        if self.config["evaluation_games"] and self.completed % eval_every == 0:
            self.phase = "evaluating"
            self.status()
            result = evaluate_suite(model, self.config["evaluation_games"], self.config["evaluation_seed"],
                                    self.config, self.opponent_policy)
            write_json(self.experiment / f"evaluation-{self.completed:04d}.json", result)
            log_evaluation(model.logger, self.config.get("validation_prefix", self.config.get("tensorboard_validation_prefix", "val")), result)
            model.logger.record(f"{self.config.get('validation_prefix', self.config.get('tensorboard_validation_prefix', 'val'))}/ppo_iteration", self.completed)
            if result["mean_placement"] < self.best:
                self.best = result["mean_placement"]
                save_model(model, self.best_model_path)
            metrics["evaluation_mean_placement"] = result["mean_placement"]
        with (self.experiment / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        if self.config["train_per_optimizer_step"]:
            for name, value in list(model.logger.name_to_value.items()):
                if name.startswith("train/"):
                    if name != "train/n_updates":
                        model.logger.record(f"ppo_summary/{name.removeprefix('train/')}", value)
                    model.logger.record(name, value, exclude=("tensorboard", "wandb"))
        log_step = (model.optimizer_step_count() if self.config["tracking_backend"] == "wandb"
                    or self.config.get("tensorboard_step_unit") == "optimizer_step" else model.num_timesteps)
        model.logger.dump(log_step)
        self.phase = "iteration_complete"
        self.status(last_iteration=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--forward-precision", choices=FORWARD_PRECISIONS,
                        help="Neural forward precision; missing inherits config/checkpoint")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--resume", action="store_true", help="Resume a fully updated iteration checkpoint")
    stage.add_argument("--warm-start", action="store_true", help="Start a new stage from trained weights with fresh Adam")
    stage.add_argument("--continue-stage", action="store_true",
                       help="Start a new stage keeping trained weights and Adam, with fresh local counters")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("source_binding_required") and not (
        config.get("initial_model") and config.get("initial_model_sha256")
        and all(spec.get("model") and spec.get("sha256")
                for spec in config.get("opponent_pool", []))
    ):
        raise ValueError("Bind the selected previous-stage best path and SHA-256 before training")
    schedule = learning_rate_schedule(config)
    if config.get("preserve_optimizer_state") and not (args.continue_stage or args.resume):
        raise ValueError("This configuration requires --continue-stage (or --resume within that stage)")
    if not args.resume and config.get("initial_model_sha256"):
        if hashlib.sha256(args.model.read_bytes()).hexdigest() != config["initial_model_sha256"]:
            raise ValueError("Initial model differs from the configured stage source")
    if args.forward_precision is not None:
        config["forward_precision"] = args.forward_precision
    config.setdefault("micro_batch_size", DEFAULT_MICRO_BATCH_SIZE)
    config.setdefault("tracking_backend", tracking_settings(config)["backend"])
    config.setdefault("train_per_optimizer_step", config.get("tensorboard_train_per_optimizer_step", True))
    if config.get("reward_shaping") is not None:
        config["reward_shaping"] = asdict(RewardShapingConfig(**config["reward_shaping"]))
    if config["tracking_backend"] not in ("wandb", "tensorboard", "none"):
        raise ValueError("tracking_backend must be wandb, tensorboard or none")
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
    opponent_policy = load_frozen_opponents(config, args.device)
    options = dict(seed=config["seed"], threads=config["env_threads"],
                   config=config.get("simulator_config"), reward_shaping=config.get("reward_shaping"))
    driver = config.get("sampling_driver", "synchronous")
    if driver not in ("synchronous", "unified_actor"):
        raise ValueError("sampling_driver must be synchronous or unified_actor")
    if driver == "unified_actor" and opponent_policy is None:
        raise ValueError("Unified actor sampling requires a fixed opponent policy")
    mixed_class = ActorOpponentVectorEnv if driver == "unified_actor" else MixedOpponentVectorEnv
    env = (mixed_class(config["n_envs"], opponent_policy=opponent_policy,
                                  frozen_opponents=config.get("frozen_opponents", 4), **options)
           if opponent_policy is not None else NativeVectorEnv(config["n_envs"], **options))
    model = EntityPPO.load(
        args.model, env=env, device=args.device,
        learning_rate=schedule,
        n_steps=config["n_steps"], batch_size=config["batch_size"], n_epochs=config["n_epochs"],
        micro_batch_size=config["micro_batch_size"], rollout_buffer_class=CompactPlacementBuffer,
        rollout_buffer_kwargs={}, seed=config["seed"],
    )
    configure_precision(model, config, opponent_policy)
    total_decisions = config["iterations"] * expected
    if args.warm_start or args.continue_stage:
        config["warm_start_source_timesteps"] = model.num_timesteps
        config["warm_start_source_optimizer_steps"] = model.optimizer_step_count()
        config["stage_source_adam_total_steps"] = model.optimizer_total_step_count()
        old_budget = getattr(model, "experiment_config", {}).get("decisions_per_iteration")
        if args.continue_stage and old_budget and model.num_timesteps % old_budget:
            raise ValueError("Continuation requires a fully updated source PPO iteration")
        model.start_new_stage(preserve_adam=args.continue_stage)
        config["preserve_optimizer_state"] = args.continue_stage
        config["stage_initialization"] = (
            "inherited network, text and Adam moments/clock; fresh stage counters"
            if args.continue_stage else "inherited network and text; fresh Adam and stage counters")
    if not args.resume and model.num_timesteps != 0:
        raise ValueError("This experiment must start from the zero-step initialization")
    if model.num_timesteps % expected:
        raise ValueError("Resume requires a fully updated iteration checkpoint, not an interrupted rollout")
    if args.resume:
        prior_path = args.experiment_dir / "resolved-config.json"
        prior_config = json.loads(prior_path.read_text()) if prior_path.exists() else {}
        config["best_model_filename"] = prior_config.get(
            "best_model_filename", model.experiment_config.get("best_model_filename", "best_model.zip"))
        for name in ("stage_source_adam_total_steps", "warm_start_source_timesteps",
                     "warm_start_source_optimizer_steps"):
            if name in prior_config:
                config[name] = prior_config[name]
            elif name in model.experiment_config:
                config[name] = model.experiment_config[name]
    else:
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
        config["best_model_filename"] = f"{stamp}-best_model.zip"
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
    tracker = None
    formats = ["stdout", "csv"]
    if config["tracking_backend"] == "tensorboard":
        formats.append("tensorboard")
    logger = configure(str(log_directory), formats)
    if config["tracking_backend"] == "wandb":
        tracker = WandbTracker(args.experiment_dir, config)
        logger.output_formats.append(WandbOutputFormat(tracker))
    model.set_logger(logger)
    model.tensorboard_log = str(args.experiment_dir / "tensorboard") if config["tracking_backend"] == "tensorboard" else None
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
    recorder = Recorder(config, args.experiment_dir, args.run_dir, opponent_policy)
    recorder.init_callback(model)
    recorder.completed = initial_iteration
    recorder.started -= previous_elapsed
    if args.resume:
        metrics_path = args.experiment_dir / "metrics.jsonl"
        if metrics_path.exists():
            prior_metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
            recorder.iteration_seconds = [row["rollout_seconds"] + row["update_seconds"] for row in prior_metrics
                                          if row["iteration"] <= initial_iteration]
        # Independent heldout results must never affect validation selection on resume.
        for evaluation_path in args.experiment_dir.glob("evaluation-[0-9][0-9][0-9][0-9].json"):
            evaluation = json.loads(evaluation_path.read_text())
            recorder.best = min(recorder.best, evaluation.get("mean_placement", float("inf")))
    model._after_update = recorder.after_update
    adam_writer = None
    if config["train_per_optimizer_step"]:
        if config["tracking_backend"] == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            adam_writer = SummaryWriter(str(log_directory), filename_suffix=".adam", flush_secs=5)

        def record_optimizer_step(current, metrics, step):
            values = {f"train/{name}": value for name, value in metrics.items()}
            values.update({"train/learning_rate": current.policy.optimizer.param_groups[0]["lr"],
                           "train/ppo_iteration": current.num_timesteps // expected,
                           "train/sampled_decisions": current.num_timesteps,
                           "train/adam_total_steps": current.optimizer_total_step_count()})
            if tracker is not None:
                tracker.log(values, step)
            if adam_writer is not None:
                for name, value in values.items():
                    adam_writer.add_scalar(name, value, step)

        model._after_optimizer_step = record_optimizer_step
    if not args.resume:
        save_model(model, args.run_dir / "starting_model.zip")
    write_json(args.experiment_dir / "process.json", {"pid": os.getpid(), "started_at": now()})

    def stop(signum, frame):
        write_json(args.experiment_dir / "last-interruption-signal.json", {
            "received_at": now(), "pid": os.getpid(), "signal": signal.Signals(signum).name,
            "signal_number": signum, "sender": "not available from Python signal handler",
        })
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    status = "completed"
    try:
        if config["initial_evaluation_games"] and not args.resume:
            recorder.phase = "initial_evaluation"
            recorder.status()
            result = evaluate_suite(model, config["initial_evaluation_games"], config["evaluation_seed"],
                                    config, opponent_policy)
            recorder.best = result["mean_placement"]
            write_json(args.experiment_dir / "evaluation-0000.json", result)
            log_evaluation(model.logger, config.get("baseline_prefix", config.get("tensorboard_baseline_prefix", "val")), result)
            model.logger.dump(0)
            save_model(model, recorder.best_model_path)
        model.learn(remaining_decisions, callback=recorder,
                    reset_num_timesteps=False, log_interval=None)
        save_model(model, args.run_dir / "final_model.zip")
        if config["final_evaluation_games"]:
            write_json(args.experiment_dir / "evaluation-final-heldout.json", evaluate_suite(
                model, config["final_evaluation_games"], config["final_evaluation_seed"], config, opponent_policy))
    except KeyboardInterrupt:
        status = "interrupted"
        save_model(model, args.run_dir / "interrupted_model.zip")
    except BaseException:
        status = "failed"
        failure = getattr(getattr(env, "sampler", None), "last_failure", None)
        if failure is not None:
            write_json(args.experiment_dir / "native-action-failure.json", failure)
        save_model(model, args.run_dir / "failed_model.zip")
        raise
    finally:
        result = recorder.status(status=status,
                                 peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None)
        write_json(args.experiment_dir / "training.json", result)
        if adam_writer is not None:
            adam_writer.close()
        model.rollout_buffer.close()
        env.close()
        if tracker is not None:
            tracker.finish(exit_code=0 if status == "completed" else 1)


if __name__ == "__main__":
    main()
