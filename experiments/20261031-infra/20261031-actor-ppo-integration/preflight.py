"""One full production-size PPO iteration on disposable in-memory weights."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from taverngym.learning.actor_sampling import ActorOpponentVectorEnv
from taverngym.learning.buffer import CompactPlacementBuffer
from taverngym.learning.experiment import load_frozen_policy, now, write_json
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.schedule import LinearLearningRate

FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / "pyproject.toml").exists())
SOURCE = ROOT / "runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip"

def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    fixed = load_frozen_policy(SOURCE, "cuda", forward_precision="bf16")
    fixed_versions = [p._version for p in fixed.parameters()]
    env = ActorOpponentVectorEnv(256, seed=43, threads=4, opponent_policy=fixed, frozen_opponents=2,
        config={"turn_action_caps": [60]}, reward_shaping={"operation_cost": .0001, "cycle_cost": .0002})
    model = EntityPPO.load(SOURCE, env=env, device="cuda", n_steps=1024, batch_size=8192,
        micro_batch_size=1024, n_epochs=1, seed=43, rollout_buffer_class=CompactPlacementBuffer,
        rollout_buffer_kwargs={}, learning_rate=LinearLearningRate(2.9703e-4, 2.9703e-4))
    model.set_forward_precision("bf16")
    model.start_new_stage()
    updates = []
    def updated(current, metrics, step):
        assert all(np.isfinite(v) for v in metrics.values())
        assert all(p.grad.dtype == torch.float32 for p in current.policy.parameters() if p.grad is not None)
        updates.append({"step": step, **metrics})
    model._after_optimizer_step = updated
    started = time.monotonic()
    try:
        model.learn(262144, log_interval=None)
        assert model.num_timesteps == 262144 and model.optimizer_step_count() == len(updates) == 32
        assert fixed_versions == [p._version for p in fixed.parameters()]
        assert not fixed.optimizer.state and all(p.grad is None for p in fixed.parameters())
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
        assert np.isfinite(model.rollout_buffer.advantages).all() and model._rollout_places
        result = {"status": "passed", "at": now(), "seconds": time.monotonic() - started,
            "sampling_seconds": model.last_rollout_seconds, "update_seconds": model.last_train_seconds,
            "samples": model.num_timesteps, "adam_updates": len(updates),
            "optimizer_lr": model.policy.optimizer.param_groups[0]["lr"],
            "games_finished": len(model._rollout_places), "mean_placement": float(np.mean(model._rollout_places)),
            "known_placement_fraction": float((model.rollout_buffer.placement_targets >= 0).mean()),
            "rollout_statistics": {k: float(v) for k, v in model.logger.name_to_value.items()
                if k.startswith(("rollout/", "system/"))},
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "source_sha256": digest, "source_unchanged": True, "fixed_v1_unchanged": True,
            "new_models_saved": 0, "disposable_preflight_updates": True, "updates": updates}
        write_json(FOLDER / "full-ppo-preflight.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("updates", "rollout_statistics")}), flush=True)
    except BaseException:
        if env.sampler is not None and env.sampler.last_failure:
            write_json(FOLDER / "preflight-native-failure.json", env.sampler.last_failure)
        raise
    finally:
        model.rollout_buffer.close()
        env.close()

if __name__ == "__main__":
    main()
