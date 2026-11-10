"""A full rollout is not a completed optimization/checkpoint boundary."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import taverngym as tg
from taverngym.learning.checkpoint_completion import ensure_complete_ppo_checkpoint
from taverngym.learning.model import ModelConfig
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.vector_env import NativeVectorEnv


def small_model():
    torch.set_num_threads(2)
    env = NativeVectorEnv(2, seed=272, threads=1, config={"turn_action_caps": [3]})
    model = EntityPPO(EntityPolicy, env, n_steps=8, batch_size=8, micro_batch_size=4,
                      n_epochs=1, seed=272, device="cpu", policy_kwargs={"model_config":
                      asdict(ModelConfig(d_model=16, n_layers=1, n_heads=2,
                                         ff_dim=32, text_dim=8))})
    text = torch.randn(tg.N_CARDS, 8)
    text[0].zero_()
    model.policy.network.set_text_embeddings(text)
    return model


def sidecar(path, model, **changes):
    value = {"complete_PPO_boundary": True, "train_returned": True,
             "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "samples": model.num_timesteps, "stage_Adam": model.optimizer_step_count(),
             "source_Adam_offset": getattr(model, "_optimizer_step_offset", 0)}
    value.update(changes)
    Path(str(path) + ".boundary.json").write_text(json.dumps(value))


def test_full_rollout_partial_adam_interrupt_persists_and_cannot_be_overridden(tmp_path):
    model = small_model()

    original_step = model.policy.optimizer.step
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        assert model.num_timesteps == 16
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupt before second actual Adam step")
        return original_step(*args, **kwargs)

    model.policy.optimizer.step = interrupt
    try:
        with pytest.raises(RuntimeError, match="second actual Adam"):
            model.learn(16, log_interval=None)
        assert model.num_timesteps % 16 == 0 and model.optimizer_step_count() == 1
        path = tmp_path / "interrupted.zip"
        model.save(path)
        restored = EntityPPO.load(path, device="cpu")
        assert restored._ppo_completion["status"] == "in_progress"
        sidecar(path, restored)
        with pytest.raises(ValueError, match="in progress"):
            ensure_complete_ppo_checkpoint(restored, path)
    finally:
        model.env.close()


def test_success_callback_sees_completed_marker_roundtrip_and_new_stage(tmp_path):
    model = small_model()
    path = tmp_path / "complete.zip"
    seen = []

    def complete(current):
        assert current._ppo_completion["status"] == "complete"
        current.save(path)
        seen.append(ensure_complete_ppo_checkpoint(current, path))

    model._after_update = complete
    try:
        model.learn(16, log_interval=None)
        assert len(seen) == 1 and seen[0]["stage_Adam"] == 2
        restored = EntityPPO.load(path, device="cpu")
        assert ensure_complete_ppo_checkpoint(restored, path)["complete"]
        restored.start_new_stage(preserve_adam=True)
        fresh = tmp_path / "new-stage.zip"
        restored.save(fresh)
        row = ensure_complete_ppo_checkpoint(restored, fresh)
        assert row["num_timesteps"] == row["stage_Adam"] == 0
        assert row["source_Adam_offset"] == row["total_Adam"] == 2
    finally:
        model.env.close()


def test_early_KL_complete_is_not_guessed_from_fixed_adam_budget(tmp_path):
    model = small_model()
    # A deliberately tiny/negative fixture threshold takes the real KL-stop path
    # before an Adam step; returning from train still proves a complete update.
    model.target_kl = -1
    try:
        model.learn(16, log_interval=None)
        assert model.num_timesteps == 16 and model.optimizer_step_count() == 0
        path = tmp_path / "early-kl.zip"
        model.save(path)
        restored = EntityPPO.load(path, device="cpu")
        assert ensure_complete_ppo_checkpoint(restored, path)["stage_Adam"] == 0
    finally:
        model.env.close()


@pytest.fixture
def legacy(tmp_path):
    model = small_model()
    try:
        model.learn(16, log_interval=None)
        del model._ppo_completion
        path = tmp_path / "legacy.zip"
        model.save(path)
        restored = EntityPPO.load(path, device="cpu")
        # Constructor default must not accidentally authorize this legacy ZIP.
        assert restored._ppo_completion["num_timesteps"] == 0
        return path, restored
    finally:
        model.env.close()


def test_positive_legacy_requires_and_accepts_exact_sidecar(legacy):
    path, model = legacy
    with pytest.raises(ValueError, match="requires SHA-bound"):
        ensure_complete_ppo_checkpoint(model, path)
    sidecar(path, model)
    assert ensure_complete_ppo_checkpoint(model, path)["evidence"] == "SHA_bound_legacy_boundary"


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "0" * 64), ("samples", 32),
    ("stage_Adam", 3), ("source_Adam_offset", 1), ("train_returned", False),
    ("complete_PPO_boundary", False)])
def test_legacy_sidecar_drift_refused(legacy, field, value):
    path, model = legacy
    sidecar(path, model, **{field: value})
    with pytest.raises(ValueError, match="boundary SHA/samples/Adam/completion"):
        ensure_complete_ppo_checkpoint(model, path)


def test_legacy_zero_initialization_allowed_but_inprogress_never_allowed(tmp_path):
    model = small_model()
    try:
        del model._ppo_completion
        path = tmp_path / "zero.zip"
        model.save(path)
        restored = EntityPPO.load(path, device="cpu")
        assert ensure_complete_ppo_checkpoint(restored, path)["evidence"] == "zero_step_initialization"
        restored._ppo_completion["status"] = "in_progress"
        sidecar(path, restored)
        with pytest.raises(ValueError, match="in progress"):
            ensure_complete_ppo_checkpoint(restored, path)
    finally:
        model.env.close()
