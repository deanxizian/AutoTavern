"""Verify the completed first version without sampling or updating its weights."""

import hashlib
import json
import math
from pathlib import Path

import torch

from taverngym.learning.ppo import EntityPPO
from taverngym.learning.schedule import LinearLearningRate


def main():
    torch.set_num_threads(2)
    folder = Path(__file__).resolve().parent
    models = folder.parent.parent / "runs" / folder.name
    source = models.parent / "20261031-ppo-small-init/initial_model.zip"
    initial_hash = "3c8f024a0cfb1526298d0d9a105d4d5a2e283cc724eb0682b160f0a9a7d38e53"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == initial_hash
    progress = json.loads((folder / "progress.json").read_text())
    assert progress["status"] == "completed"
    config = json.loads((folder / "config.json").read_text())
    metrics = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 100
    schedule = LinearLearningRate(config["learning_rate_start"], config["learning_rate_end"])
    for index, record in enumerate(metrics, start=1):
        assert record["iteration"] == index and record["decisions"] == index * 8192
        assert math.isclose(record["train/learning_rate"], schedule(1 - index / 100), rel_tol=1e-10)
        assert all(math.isfinite(value) for value in record.values() if isinstance(value, (int, float)))
    if not (models / "final_model.zip").exists():
        retained = json.loads((folder / "retention.json").read_text())
        result = json.loads((folder / "verification.json").read_text())
        assert retained["status"] == "completed" and result["training_completed"]
        removed_final = next(item for item in retained["removed_models"] if item["path"].endswith("/final_model.zip"))
        assert removed_final["sha256"] == result["final_model_sha256"]
        best = models / "best_model.zip"
        assert hashlib.sha256(best.read_bytes()).hexdigest() == retained["kept_model"]["sha256"]
        model = EntityPPO.load(best, device="cpu")
        assert model.num_timesteps == retained["kept_model"]["samples"]
        assert int(model.policy.optimizer.state[model.policy.network.identity.weight]["step"].item()) == retained["kept_model"]["training_updates"]
        assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
        result.update(full_training_checkpoint_verified_before_cleanup=True,
                      full_training_checkpoint_retained=False, retained_best_model=str(best),
                      retained_best_training_updates=retained["kept_model"]["training_updates"],
                      retained_best_verified_after_cleanup=True)
        (folder / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return
    model = EntityPPO.load(models / "final_model.zip", device="cpu")
    initial = EntityPPO.load(source, device="cpu")
    assert model.num_timesteps == 819200 and model._n_updates == 100
    identity = model.policy.network.identity.weight
    assert int(model.policy.optimizer.state[identity]["step"].item()) == 100
    assert model.micro_batch_size == 1024 and model.batch_size == 8192 and model.n_epochs == 1
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
    assert torch.equal(model.policy.network.text_features, initial.policy.network.text_features)
    assert not torch.equal(model.policy.network.identity.weight, initial.policy.network.identity.weight)
    assert initial.num_timesteps == 0 and not initial.policy.optimizer.state
    assert math.isclose(model.policy.optimizer.param_groups[0]["lr"], 3e-6, rel_tol=1e-10)
    steps = sorted(set(int(state["step"].item()) for state in model.policy.optimizer.state.values()))
    result = {
        "status": "passed", "training_completed": True,
        "samples": model.num_timesteps, "effective_minibatches": 100,
        "adam_updates_verified_on_shared_identity": 100, "adam_parameter_step_values": steps,
        "micro_batch_size": model.micro_batch_size, "gradient_accumulation": 8,
        "microbatch_backwards": 800, "learning_rate_initial_configured": 3e-4,
        "learning_rate_first_update": metrics[0]["train/learning_rate"],
        "learning_rate_final": model.policy.optimizer.param_groups[0]["lr"],
        "linear_schedule_matches_all_updates": True, "parameters_finite": True,
        "source_initialization_unchanged": True, "rule_text_embeddings_unchanged": True,
        "final_model": str(models / "final_model.zip"),
        "final_model_sha256": hashlib.sha256((models / "final_model.zip").read_bytes()).hexdigest(),
        "optimizer_restored_from_final_checkpoint": True,
    }
    (folder / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
