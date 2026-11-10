"""The generic CLI keeps the checkpoint's rules and schema during real updates."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")


@pytest.mark.parametrize("mode", ["updated", "zero-adam", "interrupted"])
def test_generic_train_inherits_simulator_and_emits_schema_v2(tmp_path, monkeypatch, mode):
    import taverngym as tg
    from taverngym.learning import train
    from taverngym.learning.checkpoint_completion import ensure_complete_ppo_checkpoint
    from taverngym.learning.metric_schema_v2 import SCHEMA_PATH
    from taverngym.learning.model import ModelConfig
    from taverngym.learning.policy import EntityPolicy
    from taverngym.learning.ppo import EntityPPO
    from taverngym.learning.vector_env import NativeVectorEnv

    simulator = {"hero_choices": 2, "turn_action_caps": [60]}
    config = {"model_version": "v8", "metric_schema_version": 2,
              "metric_schema_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
              "n_envs": 2, "seed": 4, "simulator_config": simulator,
              "forward_precision": "fp32", "tracking": {"mode": "offline", "entity": None}}
    env = NativeVectorEnv(2, seed=4, threads=1, config=simulator)
    model = EntityPPO(EntityPolicy, env, n_steps=4, batch_size=8, micro_batch_size=4,
                      n_epochs=1, seed=4, device="cpu", policy_kwargs={
                          "model_config": asdict(ModelConfig(d_model=32, n_layers=1,
                                                             n_heads=4, ff_dim=64, text_dim=16))})
    text = torch.randn(tg.N_CARDS, 16)
    text[0].zero_()
    model.policy.network.set_text_embeddings(text)
    model.experiment_config = config
    if mode == "zero-adam":
        # Exercise two successful KL-stopped PPOs sharing Adam step zero.
        model.target_kl = -1
    source = tmp_path / "initial.zip"
    model.save(source)
    env.close()
    sent, rules = [], []
    run = SimpleNamespace(entity=None, project="test", id="test-v8", url="offline",
                          define_metric=lambda *a, **k: None, log=sent.append,
                          finish=lambda **k: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **k: run,
                                                             Settings=lambda **k: k))
    original = train.make_vector_env

    def capture(*args, **kwargs):
        rules.append(kwargs["config"])
        return original(*args, **kwargs)

    monkeypatch.setattr(train, "make_vector_env", capture)
    if mode == "interrupted":
        collect = EntityPPO.collect_rollouts
        calls = 0

        def interrupted(current, *args, **kwargs):
            nonlocal calls
            result = collect(current, *args, **kwargs)
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("interrupt second PPO before optimization")
            return result

        monkeypatch.setattr(EntityPPO, "collect_rollouts", interrupted)
    output, records = tmp_path / "models", tmp_path / "records"
    monkeypatch.setattr(sys, "argv", ["train", "--model", str(source), "--run-dir", str(output),
        "--experiment-dir", str(records), "--total-timesteps", "16", "--device", "cpu",
        "--micro-batch-size", "4", "--checkpoint-every", "5"])
    train.main()
    assert rules == [simulator]
    rows = [json.loads(line) for line in (records / "wandb-metrics.jsonl").read_text().splitlines()]
    assert rows == sent
    adam_steps = {"updated": [1, 2], "zero-adam": [], "interrupted": [1]}[mode]
    assert [row["optimizer_step"] for row in rows if "train/optimization_adam_loss" in row] == adam_steps
    assert not any("train/loss" in row or "system/rollout_seconds" in row for row in rows)
    iterations = [row["train/progress_ppo_iteration"] for row in rows
                  if "train/performance_rollout_seconds" in row]
    assert iterations == ([1] if mode == "interrupted" else [1, 2])
    filename = "interrupted_model.zip" if mode == "interrupted" else "final_model.zip"
    restored = EntityPPO.load(output / filename, device="cpu")
    assert restored.optimizer_step_count() == len(adam_steps)
    assert restored.experiment_config["simulator_config"] == simulator
    periodic = sorted((output / "checkpoints").glob("*.zip"))
    assert {p.name for p in periodic} == ({"ppo-small_8_steps.zip"} if mode == "interrupted"
        else {"ppo-small_8_steps.zip", "ppo-small_16_steps.zip"})
    for path in periodic:
        checkpoint = EntityPPO.load(path, device="cpu")
        assert ensure_complete_ppo_checkpoint(checkpoint, path)["complete"]
        assert checkpoint.num_timesteps % 8 == 0
    if mode == "interrupted":
        with pytest.raises(ValueError, match="in progress"):
            ensure_complete_ppo_checkpoint(restored, output / filename)
        # The last periodic save remains a usable recovery source even though
        # the newer diagnostic archive contains a complete but unoptimized rollout.
        monkeypatch.setattr(EntityPPO, "collect_rollouts", collect)
        monkeypatch.setattr(sys, "argv", ["train", "--model", str(periodic[0]),
            "--run-dir", str(tmp_path / "recovered"), "--experiment-dir",
            str(tmp_path / "recovery-records"), "--total-timesteps", "8",
            "--micro-batch-size", "4", "--device", "cpu"])
        train.main()
        recovered_path = tmp_path / "recovered/final_model.zip"
        recovered = EntityPPO.load(recovered_path, device="cpu")
        assert ensure_complete_ppo_checkpoint(recovered, recovered_path)["complete"]
        assert recovered.num_timesteps == 16
        assert recovered.optimizer_step_count() == 2
