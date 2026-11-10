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


def test_generic_train_inherits_simulator_and_emits_schema_v2(tmp_path, monkeypatch):
    import taverngym as tg
    from taverngym.learning import train
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
    output, records = tmp_path / "models", tmp_path / "records"
    monkeypatch.setattr(sys, "argv", ["train", "--model", str(source), "--run-dir", str(output),
        "--experiment-dir", str(records), "--total-timesteps", "16", "--device", "cpu",
        "--micro-batch-size", "4"])
    train.main()
    assert rules == [simulator]
    rows = [json.loads(line) for line in (records / "wandb-metrics.jsonl").read_text().splitlines()]
    assert rows == sent
    assert [row["optimizer_step"] for row in rows if "train/optimization_adam_loss" in row] == [1, 2]
    assert not any("train/loss" in row or "system/rollout_seconds" in row for row in rows)
    restored = EntityPPO.load(output / "final_model.zip", device="cpu")
    assert restored.optimizer_step_count() == 2
    assert restored.experiment_config["simulator_config"] == simulator
