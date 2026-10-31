"""One disposable full-size rollout/update before the continued V3 stage."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

import taverngym as tg
from taverngym.learning.actor_sampling import ActorOpponentVectorEnv
from taverngym.learning.buffer import CompactPlacementBuffer
from taverngym.learning.experiment import load_frozen_opponents, now, write_json
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.precision import configure_precision
from taverngym.learning.schedule import learning_rate_schedule


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
FOLDER = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "experiments/20261031-exp03-v2-continuation/20261031-ppo-v3-2v1-3v2-warmup"


class Progress(BaseCallback):
    def __init__(self):
        super().__init__()
        self.last = time.monotonic()

    def _on_step(self):
        if time.monotonic() - self.last > 10:
            print("Preflight learner samples:", self.model.num_timesteps, flush=True)
            self.last = time.monotonic()
        return True


def main():
    config = json.loads((EXPERIMENT / "config.json").read_text())
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported() and tg.build_is_current()
    torch.cuda.reset_peak_memory_stats()
    frozen = load_frozen_opponents(config, "cuda")
    snapshots = {group.name: {name: value.clone() for name, value in group.policy.state_dict().items()}
                 for group in frozen.groups}
    env = ActorOpponentVectorEnv(config["n_envs"], seed=config["seed"], threads=config["env_threads"],
        opponent_policy=frozen, frozen_opponents=config["frozen_opponents"],
        config=config["simulator_config"], reward_shaping=config["reward_shaping"])
    source = ROOT / config["initial_model"]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == config["initial_model_sha256"]
    model = EntityPPO.load(source, env=env, device="cuda", learning_rate=learning_rate_schedule(config),
        n_steps=config["n_steps"], batch_size=config["batch_size"], n_epochs=config["n_epochs"],
        micro_batch_size=config["micro_batch_size"], rollout_buffer_class=CompactPlacementBuffer,
        rollout_buffer_kwargs={}, seed=config["seed"])
    configure_precision(model, config, frozen)
    assert model.optimizer_total_step_count() == 3200
    model.start_new_stage(preserve_adam=True)
    assert model.optimizer_step_count() == model.num_timesteps == 0
    assert all(group.policy.network.identity.weight.data_ptr() != model.policy.network.identity.weight.data_ptr()
               for group in frozen.groups)
    pointer = model.policy.network.pointer[-1].weight.detach().clone()
    budget = config["iterations"] * config["decisions_per_iteration"]
    _, callback = model._setup_learn(budget, Progress(), reset_num_timesteps=False)
    started = time.monotonic()
    try:
        assert model.collect_rollouts(env, callback, model.rollout_buffer, config["n_steps"])
        assert model.num_timesteps == 131072
        assert all(p.grad is None for p in model.policy.parameters())
        model._update_current_progress_remaining(model.num_timesteps, budget)
        model.train()
        assert model.optimizer_step_count() == 16 and model.optimizer_total_step_count() == 3216
        expected_lr = learning_rate_schedule(config)(1 - 1 / config["iterations"])
        assert abs(model.policy.optimizer.param_groups[0]["lr"] - expected_lr) < 1e-15
        assert not torch.equal(pointer, model.policy.network.pointer[-1].weight)
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        for group in frozen.groups:
            assert all(torch.equal(value, snapshots[group.name][name]) for name, value in group.policy.state_dict().items())
            assert not group.policy.optimizer.state
            assert not any(p.requires_grad or p.grad is not None for p in group.policy.parameters())
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        result = {"status": "passed", "at": now(), "seconds": time.monotonic() - started,
                  "samples": model.num_timesteps, "stage_adam_updates": model.optimizer_step_count(),
                  "adam_total_steps": model.optimizer_total_step_count(), "optimizer_lr": model.policy.optimizer.param_groups[0]["lr"],
                  "sampling_seconds": model.last_rollout_seconds, "update_seconds": model.last_train_seconds,
                  "games_finished": len(model._rollout_places), "opponents": "2 fixed V1 + 3 fixed V2 + 2 scripted",
                  "learner_mode": "stochastic BF16", "n_envs": 128, "steps_per_env": 1024,
                  "fixed_policies_unchanged": True, "sampling_retained_no_gradients": True,
                  "source_unchanged": True, "source_sha256": source_hash,
                  "target_ppo_iterations": config["iterations"],
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "known_placement_fraction": float(np.mean(model.rollout_buffer.placement_targets >= 0)),
                  "new_models_saved": 0, "formal_training_started": False, "disposable_updates": True,
                  "actor_metrics": {name: float(value) for name, value in model.logger.name_to_value.items()
                                    if name.startswith("system/actor_")}}
        write_json(FOLDER / "full-ppo-preflight.json", result)
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
