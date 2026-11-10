"""Bounded real-GPU rehearsal; discard probe weights and keep measurements only."""

import hashlib
import json
from pathlib import Path
import time

import torch

from stable_baselines3.common.callbacks import BaseCallback
from taverngym.learning.buffer import CompactPlacementBuffer
from taverngym.learning.experiment import load_frozen_policy, now, write_json
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.schedule import LinearLearningRate
from taverngym.learning.vector_env import MixedOpponentVectorEnv


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
FOLDER = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip"


def weights(policy):
    digest = hashlib.sha256()
    for name, value in policy.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    fixed = load_frozen_policy(SOURCE, "cuda")
    original = weights(fixed)
    env = MixedOpponentVectorEnv(256, seed=43, threads=4, opponent_policy=fixed,
                                config={"turn_action_caps": [60]},
                                reward_shaping={"operation_cost": 0.0001, "cycle_cost": 0.0002})
    model = EntityPPO.load(SOURCE, env=env, device="cuda", n_steps=128, batch_size=8192,
                           n_epochs=1, micro_batch_size=1024, seed=43,
                           learning_rate=LinearLearningRate(3e-4, 2.9703e-4),
                           rollout_buffer_class=CompactPlacementBuffer, rollout_buffer_kwargs={})
    model.start_new_stage()
    assert model.optimizer_step_count() == model.num_timesteps == 0
    assert weights(model.policy) == original

    class Progress(BaseCallback):
        def _on_step(self):
            if self.n_calls % 16 == 0:
                write_json(FOLDER / "progress.json", {"status": "running", "updated_at": now(),
                    "learner_samples": self.model.num_timesteps,
                    "opponent_decisions": env.opponent_decisions,
                    "elapsed_seconds": time.monotonic() - started})
            return True

    try:
        model.learn(32768, callback=Progress(), log_interval=None, reset_num_timesteps=False)
        assert model.num_timesteps == 32768
        assert model.optimizer_step_count() > 0
        assert weights(fixed) == original and weights(model.policy) != original
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
        metrics = {k: float(v) for k, v in model.logger.name_to_value.items()}
        result = {"status": "passed", "verified_at": now(), "source_sha256": source_hash,
                  "n_envs": 256, "n_steps": 128, "fresh_samples": model.num_timesteps,
                  "batch_size": 8192, "micro_batch_size": 1024,
                  "optimizer_steps": model.optimizer_step_count(),
                  "sampling_seconds": model.last_rollout_seconds, "update_seconds": model.last_train_seconds,
                  "opponent_decisions": env.opponent_decisions, "waiting_rounds": env.waiting_rounds,
                  "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "max_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "all_weights_finite": True, "frozen_opponent_unchanged": True,
                  "warm_start_weights_exact": True, "adam_initially_empty": True,
                  "no_probe_checkpoint_saved": True, "metrics": metrics,
                  "estimated_training_iterations_seconds": 100 * 8 * (
                      model.last_rollout_seconds + model.last_train_seconds)}
        write_json(FOLDER / "results.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2), flush=True)
    finally:
        model.rollout_buffer.close()
        env.close()


if __name__ == "__main__":
    main()
