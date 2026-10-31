"""Forward precision is shared by rollout/update; optimizer and storage remain FP32."""

from dataclasses import asdict
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

import taverngym as tg
from taverngym.learning.experiment import load_frozen_policy
from taverngym.learning.model import ModelConfig
from taverngym.learning.policy import EntityPolicy
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.precision import configure_precision
from taverngym.learning.vector_env import MixedOpponentVectorEnv, NativeVectorEnv


def model_for(precision="fp32", device="cpu"):
    torch.set_num_threads(2)
    env = NativeVectorEnv(2, seed=4, threads=1, config={"turn_action_caps": [3]})
    model = EntityPPO(EntityPolicy, env, n_steps=8, batch_size=8, micro_batch_size=4, n_epochs=1,
                      seed=4, device=device, policy_kwargs={"forward_precision": precision,
                      "model_config": asdict(ModelConfig(d_model=32, n_layers=1, n_heads=4, ff_dim=64, text_dim=16))})
    text = torch.randn(tg.N_CARDS, 16)
    text[0].zero_()
    model.policy.network.set_text_embeddings(text.to(device))
    return model


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_forward_paths_have_native_masks_and_fp32_outputs(precision):
    model = model_for(precision)
    policy = model.policy
    obs, _ = policy.obs_to_tensor(model.env.reset())
    raw_dtypes = []
    hook = policy.network.pointer[-1].register_forward_hook(lambda m, a, out: raw_dtypes.append(out.dtype))
    try:
        policy.set_training_mode(True)
        actions, values, logp = policy(obs)
        rescore_values, rescore_logp, entropy = policy.evaluate_actions(obs, actions)
        torch.testing.assert_close(values, rescore_values, rtol=0, atol=0)
        torch.testing.assert_close(logp, rescore_logp, rtol=0, atol=0)
        assert policy.placement_logits.dtype == torch.float32
        assert all(v.dtype == torch.float32 for v in (values, logp, rescore_values, rescore_logp, entropy))
        assert obs["action_mask"][torch.arange(len(actions)), actions].all()
        policy.set_training_mode(False)
        with torch.no_grad():
            distribution = policy.get_distribution(obs)
            bootstrap = policy.predict_values(obs)
        assert bootstrap.dtype == distribution.distribution.probs.dtype == torch.float32
        assert torch.count_nonzero(distribution.distribution.probs[~obs["action_mask"].bool()]) == 0
        assert raw_dtypes and set(raw_dtypes) == {torch.bfloat16 if precision == "bf16" else torch.float32}
        assert {p.dtype for p in policy.parameters()} == {torch.float32}
    finally:
        hook.remove()
        model.env.close()


def test_bf16_updates_accumulate_fp32_gradients_and_roundtrip(tmp_path):
    model = model_for("bf16")
    before = model.policy.network.pointer[-1].weight.detach().clone()
    optimizer_steps = []

    def updated(current, metrics, step):
        optimizer_steps.append(step)
        assert all(np.isfinite(v) for v in metrics.values())
        assert all(p.grad.dtype == torch.float32 for p in current.policy.parameters() if p.grad is not None)

    model._after_optimizer_step = updated
    model.learn(16, log_interval=None)
    assert optimizer_steps == [1, 2]
    assert not torch.equal(before, model.policy.network.pointer[-1].weight)
    for name in ("rewards", "values", "log_probs", "advantages", "returns"):
        assert getattr(model.rollout_buffer, name).dtype == np.float32
    assert {v.dtype for state in model.policy.optimizer.state.values()
            for v in state.values() if torch.is_tensor(v) and v.is_floating_point()} == {torch.float32}
    path = tmp_path / "bf16.zip"
    model.save(path)
    restored = EntityPPO.load(path, device="cpu")
    assert restored.policy.forward_precision == restored.policy_kwargs["forward_precision"] == "bf16"
    assert {p.dtype for p in restored.policy.parameters()} == {torch.float32}
    assert restored.optimizer_step_count() == 2
    obs = model.env.reset()
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], restored.predict(obs, deterministic=True)[0])
    model.env.close()


def test_legacy_model_can_enable_bf16_without_changing_source_or_weights(tmp_path):
    model = model_for()
    old = tmp_path / "legacy.zip"
    model.save(old)
    old_hash = hashlib.sha256(old.read_bytes()).hexdigest()
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    frozen = load_frozen_policy(old, "cpu")
    config = {"forward_precision": "bf16"}
    configure_precision(model, config, frozen)
    assert model.policy.forward_precision == frozen.forward_precision == "bf16"
    assert config["opponent_forward_precision"] == "bf16"
    assert config["precision_settings"]["learner_sampling_forward"] == "bf16"
    assert config["precision_settings"]["learner_update_forward"] == "bf16"
    for name, value in model.policy.state_dict().items():
        assert torch.equal(value, before[name])
        assert torch.equal(frozen.state_dict()[name], before[name])
    assert not frozen.optimizer.state and not any(p.requires_grad for p in frozen.parameters())
    loaded = EntityPPO.load(old, device="cpu")
    assert loaded.policy.forward_precision == "fp32"
    converted = tmp_path / "bf16-mode.zip"
    model.save(converted)
    assert EntityPPO.load(converted, device="cpu").policy.forward_precision == "bf16"
    assert hashlib.sha256(old.read_bytes()).hexdigest() == old_hash
    with pytest.raises(ValueError, match="fp32 or bf16"):
        model.set_forward_precision("fp16")
    model.env.close()


def test_one_fixed_bot_and_learner_both_use_bf16_in_batched_games(tmp_path):
    model = model_for("bf16")
    original_env = model.env
    source = tmp_path / "fixed.zip"
    model.save(source)
    frozen = load_frozen_policy(source, "cpu", forward_precision="bf16")
    before = {name: value.clone() for name, value in frozen.state_dict().items()}
    raw = {"learner": [], "opponent": []}
    hooks = [model.policy.network.pointer[-1].register_forward_hook(
                lambda m, a, out: raw["learner"].append((out.dtype, out.shape[0]))),
             frozen.network.pointer[-1].register_forward_hook(
                lambda m, a, out: raw["opponent"].append((out.dtype, out.shape[0])))]
    env = MixedOpponentVectorEnv(2, seed=4, threads=1, opponent_policy=frozen, frozen_opponents=1,
                                config={"turn_action_caps": [3]})
    try:
        model.set_env(env)
        model.learn(16, log_interval=None)
        assert env.opponent_decisions > 0 and model.optimizer_step_count() == 2
        assert all(dtype == torch.bfloat16 for values in raw.values() for dtype, batch in values)
        assert max(batch for dtype, batch in raw["opponent"]) == 2
        assert all(torch.equal(value, before[name]) for name, value in frozen.state_dict().items())
        assert not frozen.optimizer.state
    finally:
        for hook in hooks:
            hook.remove()
        env.close()
        original_env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional")
def test_cuda_bf16_rollout_rescoring_and_optimizer_are_consistent(tmp_path):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device has no BF16 support")
    model = model_for("bf16", device="cuda")
    policy = model.policy
    obs, _ = policy.obs_to_tensor(model.env.reset())
    try:
        policy.set_training_mode(False)
        with torch.no_grad():
            actions, values, old_logp = policy(obs)
            policy.set_training_mode(True)
            actual_values, actual_logp, entropy = policy.evaluate_actions(obs, actions)
        torch.testing.assert_close(old_logp, actual_logp, rtol=0, atol=1e-6)
        torch.testing.assert_close(values, actual_values, rtol=0, atol=1e-6)
        assert actual_logp.dtype == actual_values.dtype == entropy.dtype == torch.float32
        model.learn(16, log_interval=None)
        assert model.optimizer_step_count() == 2
        assert all(p.dtype == torch.float32 and (p.grad is None or p.grad.dtype == torch.float32)
                   for p in policy.parameters())
        path = tmp_path / "cuda-bf16.zip"
        model.save(path)
        restored = EntityPPO.load(path, device="cuda")
        assert restored.policy.forward_precision == "bf16"
        np.testing.assert_array_equal(model.predict(model.env._obs, deterministic=True)[0],
                                      restored.predict(model.env._obs, deterministic=True)[0])
    finally:
        model.env.close()
