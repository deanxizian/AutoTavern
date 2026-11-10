"""New finite-chain source, rollout shape, lifecycle and hard deadline boundaries."""

import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

pytest.importorskip("torch")
from taverngym.learning import stage_chain as chain
from taverngym.learning.parallel_ranking import merged_actions
from stable_baselines3.common.save_util import save_to_zip_file
import torch


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "experiments/20261101-exp07-v7-history-ranked/"
            "20261101-ppo-v7-move0005-cycle001-forceend/config.json")


def checkpoint(path, version, value=1.):
    save_to_zip_file(path, data={"num_timesteps": 0, "_optimizer_step_offset": 7200,
        "_n_updates": 0, "n_epochs": 1,
        "experiment_config": {"decisions_per_iteration": 131072, "model_version": version}},
        params={"policy": {"weight": torch.tensor([value])}, "policy.optimizer": {
            "state": {0: {"step": torch.tensor(7200.), "exp_avg": torch.tensor([.1]),
                          "exp_avg_sq": torch.tensor([.01])}},
            "param_groups": [{"lr": 1e-5, "params": [0]}]}})


def test_256_by_512_preserves_budget_and_protocol():
    cfg = json.loads(TEMPLATE.read_text())
    cfg.update(n_envs=256, n_steps=512, expected_opponent_counts={"a": 2, "b": 2, "c": 2},
               opponent_pool=[{"name": n, "count": 2} for n in "abc"])
    chain.verify_config(cfg)
    assert cfg["n_envs"] * cfg["n_steps"] == cfg["decisions_per_iteration"] == 131072
    cfg["n_steps"] = 256
    with pytest.raises(ValueError, match="rollout shape"):
        chain.verify_config(cfg)


def test_new_first_stage_ranks_and_binds_v5_before_training(tmp_path, monkeypatch):
    checkpoint(tmp_path / "v5.zip", "v5")
    previous = {"status": "completed", "model_version": "v5",
        "retained_model": chain.inspect_checkpoint(tmp_path / "v5.zip"),
        "selection": {"ppo_iteration": 0}, "reward_shaping": None,
        "config_path": "source-config.json"}
    chain.write_json(tmp_path / "v5-result.json", previous)
    folder = tmp_path / "v7"
    folder.mkdir()
    cfg = json.loads(TEMPLATE.read_text())
    cfg.update(name="v7", n_envs=256, n_steps=512)
    chain.write_json(folder / "config.json", cfg)
    chain.write_json(folder / "experiment.json", {})
    manifest = {"root": str(tmp_path), "instance_id": "new-v7", "stages": [{"version": "v7",
        "experiment_dir": "v7", "prepared_config_sha256": chain.digest(folder / "config.json"),
        "rank_opponents": True}], "initial_source_result": {"path": "v5-result.json",
        "sha256": chain.digest(tmp_path / "v5-result.json"), "model_version": "v5"},
        "budget": {"ppo_iterations": 100, "samples": 13107200, "adam_updates": 1600}}
    chain.write_json(tmp_path / "manifest.json", manifest)
    subject = chain.StageChain(tmp_path / "manifest.json")
    from taverngym.learning import stage_extension as extension
    seen = []

    def ranking(manifest, prior, known):
        assert prior == previous and known["v5"] == previous
        seen.append("ranking")
        return {"candidate_count": 4}

    def pool(root, path, prior):
        bound = chain.read_json(path / "config.json")
        bound.update(expected_opponent_counts={"bot_v5": 6}, opponent_pool=[{
            "name": "bot_v5", "count": 6, "model": "v5.zip",
            "sha256": prior["retained_model"]["sha256"]}])
        chain.write_json(path / "config.json", bound)
        return bound

    def train(stage, index, path, bound, resume=False):
        assert bound["initial_model"] == "v5.zip"
        assert bound["source_adam_total_steps"] == 7200
        seen.append("train")

    def child(command, stage, index, phase):
        if phase == "ranking":
            return
        result = {**previous, "model_version": "v7", "fresh_samples": 13107200,
                  "actual_adam_updates": 1600}
        chain.write_json(folder / "results.json", result)

    monkeypatch.setattr(extension, "ranking_input", ranking)
    monkeypatch.setattr(extension, "bind_ranked_pool", pool)
    monkeypatch.setattr(subject, "train", train)
    monkeypatch.setattr(subject, "run_child", child)
    monkeypatch.setattr(chain.signal, "signal", lambda *a: None)
    subject.run()
    assert seen == ["ranking", "train"]
    assert subject.state["status"] == "completed"


@pytest.mark.parametrize("value,status,canonical", [(1., "retired", "v5"), (2., "active", "v7")])
def test_lifecycle_compares_policy_and_complete_adam(tmp_path, value, status, canonical):
    checkpoint(tmp_path / "v5.zip", "v5")
    checkpoint(tmp_path / "v7.zip", "v7", value)
    chain.write_json(tmp_path / "registry.json", {"schema_version": 1, "versions": {
        "v5": {"status": "active", "canonical_version": "v5"}}})
    folder = tmp_path / "v7"
    folder.mkdir()
    chain.write_json(folder / "config.json", {"initial_model_version": "v5", "iterations": 100})
    result = {"model_version": "v7", "fresh_samples": 13107200,
              "source_model": chain.inspect_checkpoint(tmp_path / "v5.zip"),
              "retained_model": chain.inspect_checkpoint(tmp_path / "v7.zip")}
    chain.register_lifecycle(tmp_path, "registry.json", folder, result)
    chain.register_lifecycle(tmp_path, "registry.json", folder, result)
    entry = chain.read_json(tmp_path / "registry.json")["versions"]["v7"]
    assert (entry["status"], entry["canonical_version"]) == (status, canonical)
    assert entry["policy_and_adam_unchanged"] == (status == "retired")


def test_hard_deadline_terminates_child_and_prevents_launch_after_expiry(tmp_path, monkeypatch):
    folder = tmp_path / "v7"
    folder.mkdir()
    chain.write_json(folder / "config.json", {"name": "v7"})
    chain.write_json(tmp_path / "manifest.json", {"root": str(tmp_path), "instance_id": "clock",
        "stages": [], "budget": {"ppo_iterations": 300, "samples": 39321600,
                                 "adam_updates": 4800}})
    subject = chain.StageChain(tmp_path / "manifest.json")
    subject.state["deadline_unix"] = time.time() + .15
    real_sleep = time.sleep
    monkeypatch.setattr(chain.time, "sleep", lambda _: real_sleep(.01))
    with pytest.raises(TimeoutError):
        subject.run_child([sys.executable, "-c", "import time; time.sleep(60)"],
                          {"version": "v7", "experiment_dir": "v7"}, 0, "ranking")
    assert subject.child.poll() is not None
    assert subject.state["stop_reason"] == "wall_clock_budget_exhausted"
    with pytest.raises(TimeoutError):
        subject.run_child([sys.executable, "-c", "raise Exception('should not run')"],
                          {"version": "v7", "experiment_dir": "v7"}, 0, "ranking")


def test_candidate_and_reference_same_weights_merge_across_shards():
    class Env:
        n_slots = 7

        def observe(self):
            return {"features": np.ones((7, 2)), "cards": np.ones((7, 2)),
                    "action_mask": np.ones((7, 3), dtype=bool), "ready": np.ones(7, bool)}

    class Policy:
        def predict(self, obs, deterministic):
            assert len(obs["features"]) == 14
            assert deterministic is False
            return np.ones(14, np.int64), None

    shards = [{"env": Env(), "active": np.ones(1, bool),
               "identities": np.array(["same"] * 7)} for _ in range(2)]
    stats = {"same": []}
    merged_actions(shards, {"same": Policy()}, stats)
    assert stats == {"same": [[14, 2]]}
    assert all((s["actions"] == 1).all() for s in shards)
