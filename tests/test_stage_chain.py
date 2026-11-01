"""Finite-stage handoff, best selection, locking, and recovery failure boundaries."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")
from stable_baselines3.common.save_util import save_to_zip_file

from taverngym.learning import stage_chain as chain


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "experiments/20261031-exp04-v4-cycle-only/"
            "20261031-ppo-v4-cycle-only/config.json")


def config():
    return json.loads(TEMPLATE.read_text())


def evaluation(placement, games=64):
    conditions = {}
    for name in ("mixed_sampled", "mixed_greedy"):
        conditions[name] = {"mean_placement": float(placement), "games": games,
            "placements": [placement] * games, "seed": 40000003,
            "deterministic": name == "mixed_greedy", "opponent_groups": [
                {"name": "bot_v1", "count": 2, "deterministic": False},
                {"name": "bot_v2", "count": 3, "deterministic": False}]}
    return {**conditions["mixed_sampled"], "conditions": conditions,
            "selection_condition": "mixed_sampled"}


@pytest.mark.parametrize("placements,selected", [((3, 3, 4), 0), ((4, 3, 3), 50), ((4, 3, 2), 100)])
def test_val_selection_is_strict_earliest_and_never_heldout(tmp_path, placements, selected):
    for iteration, placement in zip((0, 50, 100), placements):
        chain.write_json(tmp_path / f"evaluation-{iteration:04d}.json", evaluation(placement))
    chain.write_json(tmp_path / "evaluation-best-heldout.json", evaluation(1))
    assert chain.validation_selection(tmp_path, config())[0][0] == selected


def test_val_requires_exact_stage_points_and_complete_games(tmp_path):
    for iteration in (0, 50):
        chain.write_json(tmp_path / f"evaluation-{iteration:04d}.json", evaluation(3))
    with pytest.raises(ValueError, match="iteration set"):
        chain.validation_selection(tmp_path, config())
    broken = evaluation(3)
    broken["conditions"]["mixed_greedy"]["placements"].pop()
    with pytest.raises(ValueError, match="game count"):
        chain.validate_evaluation(broken, 64, 40000003)


def test_instance_lock_blocks_another_process(tmp_path):
    with chain.instance_lock(tmp_path):
        result = subprocess.run([sys.executable, "-c",
            "from pathlib import Path; from taverngym.learning.stage_chain import instance_lock; "
            f"\nwith instance_lock(Path({str(tmp_path)!r})): pass"], capture_output=True, text=True)
        assert result.returncode != 0
        assert "Another controller" in result.stderr
    with chain.instance_lock(tmp_path):
        pass


def checkpoint(path, *, iteration=0, adam=5600, offset=5600, updates=None, finite=True):
    save_to_zip_file(path, data={"num_timesteps": iteration * 131072,
        "_optimizer_step_offset": offset, "_n_updates": iteration if updates is None else updates,
        "n_epochs": 1, "experiment_config": {"decisions_per_iteration": 131072}},
        params={"policy": {"weight": torch.tensor([1. if finite else float("nan")])},
                "policy.optimizer": {"state": {0: {"step": torch.tensor(float(adam)),
                    "exp_avg": torch.tensor([.1]), "exp_avg_sq": torch.tensor([.01])}},
                    "param_groups": [{"lr": 1e-5, "params": [0]}]}})


def test_checkpoint_inspects_step0_and_full_adam(tmp_path):
    path = tmp_path / "best.zip"
    checkpoint(path)
    info = chain.inspect_checkpoint(path)
    assert info["ppo_iteration"] == info["stage_optimizer_steps"] == 0
    assert info["adam_total_steps"] == 5600
    assert info["adam_states"] == 1
    assert info["learning_rates"] == [1e-5]
    assert len(info["policy_sha256"]) == len(info["adam_moments_sha256"]) == 64


@pytest.mark.parametrize("kwargs,message", [({"finite": False}, "Nonfinite policy"),
    ({"iteration": 50, "updates": 49}, "incomplete PPO")])
def test_reject_nonfinite_or_partial_optimizer_checkpoint(tmp_path, kwargs, message):
    path = tmp_path / "failed.zip"
    checkpoint(path, **kwargs)
    with pytest.raises(ValueError, match=message):
        chain.inspect_checkpoint(path)


def test_bind_step0_best_inherits_actual_adam_and_is_idempotent(tmp_path):
    source = tmp_path / "v4-best.zip"
    checkpoint(source)
    folder = tmp_path / "v5"
    folder.mkdir()
    cfg = config()
    cfg.update(initial_model=None, initial_model_sha256=None)
    chain.write_json(folder / "config.json", cfg)
    chain.write_json(folder / "experiment.json", {})
    previous = {"status": "completed", "model_version": "v4",
                "retained_model": chain.inspect_checkpoint(source),
                "selection": {"ppo_iteration": 0}}
    bound = chain.bind_source(tmp_path, folder, previous)
    assert bound["initial_model"] == "v4-best.zip"
    assert bound["initial_model_ppo_iteration"] == 0
    assert bound["source_adam_total_steps"] == 5600
    assert bound["derived_budget"]["final_total_adam_updates_max"] == 7200
    assert chain.bind_source(tmp_path, folder, previous) == bound
    changed = copy.deepcopy(previous)
    changed["retained_model"]["adam_total_steps"] += 1
    with pytest.raises(ValueError, match="checkpoint changed"):
        chain.bind_source(tmp_path, folder, changed)


def test_top3_pool_can_omit_latest_without_changing_learner_source(tmp_path):
    source = tmp_path / "v4-best.zip"
    checkpoint(source)
    folder = tmp_path / "v5"
    folder.mkdir()
    cfg = config()
    cfg.update(initial_model=None, initial_model_sha256=None, frozen_opponents=6,
               expected_opponent_counts={f"bot_v{n}": 2 for n in (1, 2, 3)})
    cfg["opponent_pool"] = [{"name": f"bot_v{n}", "count": 2,
                            "model": f"v{n}.zip", "sha256": str(n)} for n in (1, 2, 3)]
    chain.write_json(folder / "config.json", cfg)
    chain.write_json(folder / "experiment.json", {})
    previous = {"status": "completed", "model_version": "v4",
                "retained_model": chain.inspect_checkpoint(source), "selection": {"ppo_iteration": 0}}
    bound = chain.bind_source(tmp_path, folder, previous)
    assert bound["initial_model"] == "v4-best.zip"
    assert bound["initial_model_sha256"] == chain.digest(source)
    assert bound["source_adam_total_steps"] == 5600
    assert [spec["name"] for spec in bound["opponent_pool"]] == [f"bot_v{n}" for n in (1, 2, 3)]


def controller(tmp_path):
    manifest = {"root": str(tmp_path), "instance_id": "test", "stages": [],
                "budget": {"ppo_iterations": 300, "samples": 39321600, "adam_updates": 4800}}
    path = tmp_path / "manifest.json"
    chain.write_json(path, manifest)
    return chain.StageChain(path, resume=True, recovery_reason="diagnosed test failure")


def test_recovery_enforces_one_retry_and_never_resumes_a_signal(tmp_path):
    subject = controller(tmp_path)
    subject.state["recoveries"] = {"v4": 1}
    with pytest.raises(ValueError, match="limit of one"):
        subject.recover({"version": "v4"}, 0, tmp_path, config())
    subject.state["recoveries"] = {}
    (tmp_path / "last-interruption-signal.json").write_text("{}")
    with pytest.raises(ValueError, match="stop signal"):
        subject.recover({"version": "v4"}, 0, tmp_path, config())


def test_recovery_rejects_budget_loss_before_launching_process(tmp_path):
    subject = controller(tmp_path)
    cfg = config()
    models = tmp_path / "runs" / cfg["name"]
    models.mkdir(parents=True)
    checkpoint(models / "latest_model.zip", iteration=50, adam=6400)
    chain.write_json(tmp_path / "progress.json", {"sampled_decisions": 50 * 131072 + 128,
                                               "optimizer_steps": 800, "phase": "sampling"})
    with pytest.raises(ValueError, match="Discarded partial work"):
        subject.recover({"version": "v4"}, 0, tmp_path, cfg)
    assert subject.child is None


def test_boundary100_repair_does_not_train_extra_ppo(tmp_path, monkeypatch):
    subject = controller(tmp_path)
    cfg = config()
    models = tmp_path / "runs" / cfg["name"]
    models.mkdir(parents=True)
    checkpoint(models / "latest_model.zip", iteration=100, adam=7200)
    chain.write_json(tmp_path / "progress.json", {"sampled_decisions": 100 * 131072,
                                               "optimizer_steps": 1600, "phase": "evaluating"})
    launched = []
    monkeypatch.setattr(subject, "run_child", lambda command, *args: launched.append(command))
    monkeypatch.setattr(subject, "train", lambda *a, **k: pytest.fail("extra PPO training"))
    subject.recover({"version": "v4"}, 0, tmp_path, cfg)
    assert len(launched) == 1 and "repair" in launched[0]
    assert subject.state["recoveries"]["v4"] == 1


def test_no_resume_of_live_orphan_child(tmp_path):
    subject = controller(tmp_path)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)",
                              "taverngym.learning.experiment"])
    subject.previous_child_pid = child.pid
    try:
        with pytest.raises(ValueError, match="still alive"):
            subject.run()
    finally:
        child.terminate()
        child.wait()


def test_budget_includes_all_completed_stages(tmp_path):
    subject = controller(tmp_path)
    subject.record_completed({"version": "v4"}, tmp_path, {"status": "completed",
        "fresh_samples": 13107200, "actual_adam_updates": 1600,
        "retained_model": {}, "selection": {}})
    result = subject.budget(1, {"completed_iterations": 50, "sampled_decisions": 6553600,
                             "optimizer_steps": 800})
    assert result["used"] == {"ppo_iterations": 150, "samples": 19660800, "adam_updates": 2400}
    assert result["remaining"] == result["used"]


@pytest.mark.parametrize("key,value", [("evaluation_seed", 1), ("learning_rate_schedule", "warmup"),
                                     ("forward_precision", "fp32"), ("frozen_opponents", 4)])
def test_config_protocol_is_frozen(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError, match="authorized protocol|seat total"):
        chain.verify_config(cfg)


def test_missing_move_cost_only_is_compatible_with_normalized_historical_protocol():
    from taverngym.learning.rewards import reward_protocols_equal

    cfg = config()
    old = copy.deepcopy(cfg["reward_shaping"])
    cfg["reward_shaping"] = {**old, "move_cost": 0}
    chain.verify_config(cfg)
    assert reward_protocols_equal(old, cfg["reward_shaping"])
    assert "move_cost" not in old
    assert not reward_protocols_equal(None, {"move_cost": 0})
    assert not reward_protocols_equal(old, {**old, "move_cost": .0005})
    assert not reward_protocols_equal(old, {**old, "cycle_cost": .01})
    changed = {**old, "move_cost": 0}
    changed.pop("cycle_forfeit_repetitions")
    assert not reward_protocols_equal(old, changed)
    cfg["reward_shaping"]["move_cost"] = .0005
    with pytest.raises(ValueError, match="Reward/termination protocol changed"):
        chain.verify_config(cfg)


def test_evaluation_opponent_move_protocol_accepts_zero_but_rejects_nonzero():
    result = evaluation(3)
    old = {"operation_cost": 0, "cycle_cost": .01}
    pool = [{"name": "bot_v1", "count": 2, "reward_shaping": old},
            {"name": "bot_v2", "count": 3, "reward_shaping": None}]
    for row in result["conditions"].values():
        row["opponent_groups"][0]["reward_shaping"] = {**old, "move_cost": 0}
    chain.validate_evaluation(result, 64, 40000003, pool)
    result["conditions"]["mixed_greedy"]["opponent_groups"][0]["reward_shaping"]["move_cost"] = .0005
    with pytest.raises(ValueError, match="opponent termination protocol changed"):
        chain.validate_evaluation(result, 64, 40000003, pool)


@pytest.mark.parametrize("move_cost", [0, .0005])
def test_frozen_legacy_checkpoint_accepts_only_default_zero_move_cost(monkeypatch, move_cost):
    from types import SimpleNamespace
    from taverngym.learning import experiment

    old = {"operation_cost": 0, "cycle_cost": .01}
    policy = SimpleNamespace(optimizer=SimpleNamespace(state={}),
                             requires_grad_=lambda enabled: None,
                             set_training_mode=lambda enabled: None)
    frozen = SimpleNamespace(experiment_config={"reward_shaping": old, "model_version": "v5"},
                              policy=policy)
    monkeypatch.setattr(experiment.EntityPPO, "load", lambda *a, **k: frozen)
    if move_cost:
        with pytest.raises(ValueError, match="termination protocol differs"):
            experiment.load_frozen_policy("unused", "cpu", model_version="v5",
                                          reward_shaping={**old, "move_cost": move_cost})
    else:
        assert experiment.load_frozen_policy("unused", "cpu", model_version="v5",
            reward_shaping={**old, "move_cost": 0}) is policy
    assert frozen.experiment_config["reward_shaping"] == old and "move_cost" not in old


@pytest.mark.parametrize("top3", [False, True])
def test_all_three_stages_use_actual_predecessor_best_and_stop(tmp_path, monkeypatch, top3):
    subject = controller(tmp_path)
    source = tmp_path / "source.zip"
    checkpoint(source)
    for version in ("v1", "v2", "v3"):
        checkpoint(tmp_path / f"{version}.zip")
    historical = {f"v{n}": {"name": f"bot_v{n}", "count": 2, "model_version": f"v{n}",
        "model": f"v{n}.zip", "sha256": chain.digest(tmp_path / f"v{n}.zip"),
        "reward_shaping": None} for n in range(1, 4)}
    stages = []
    for version in ("v4", "v5", "v6"):
        folder = tmp_path / version
        folder.mkdir()
        cfg = config()
        cfg.update(name=version, model_version=version,
                   initial_model="source.zip" if version == "v4" else None,
                   initial_model_sha256=chain.digest(source) if version == "v4" else None)
        for spec, old_version in zip(cfg["opponent_pool"], ("v1", "v2")):
            spec.update(model=f"{old_version}.zip",
                        sha256=chain.digest(tmp_path / f"{old_version}.zip"))
        if top3:
            cfg.update(frozen_opponents=6,
                opponent_pool=list(historical.values()) if version == "v4" else [],
                expected_opponent_counts={f"bot_v{n}": 2 for n in (1, 2, 3)}
                    if version == "v4" else None)
        chain.write_json(folder / "config.json", cfg)
        chain.write_json(folder / "experiment.json", {})
        stages.append({"version": version, "experiment_dir": version,
                       "prepared_config_sha256": chain.digest(folder / "config.json"),
                       "rank_opponents": top3 and version != "v4"})
    subject.manifest["stages"] = stages
    if top3:
        subject.manifest.update(pool_selection="top3", historical_models=historical,
            ranking_protocol={"version_order": [f"v{n}" for n in range(1, 10)]})
    events = []

    def train(stage, index, folder, cfg, resume=False):
        events.append(("train", stage["version"], cfg["initial_model"]))
        models = tmp_path / "runs" / cfg["name"]
        models.mkdir(parents=True)
        checkpoint(models / "best.zip")

    def finalize(command, stage, index, phase):
        folder = tmp_path / stage["experiment_dir"]
        if phase == "ranking":
            inputs = chain.read_json(folder / "ranking-input.json")
            events.append((phase, stage["version"], inputs["candidate_count"]))
            chain.write_json(folder / "ranking-results.json", {"status": "completed",
                "pool_selection": "top3", "input_sha256": chain.digest(folder / "ranking-input.json"),
                "selected_versions": ["v1", "v2", "v3"]})
            return
        assert phase == "finalizing"
        events.append((phase, stage["version"]))
        cfg = chain.read_json(folder / "config.json")
        selected = chain.inspect_checkpoint(tmp_path / "runs" / stage["version"] / "best.zip")
        chain.write_json(folder / "results.json", {"status": "completed",
            "model_version": stage["version"],
            "reward_shaping": cfg["reward_shaping"], "opponent_pool": cfg["opponent_pool"],
            "config_path": str(folder / "config.json"),
            "fresh_samples": 13107200, "actual_adam_updates": 1600, "retained_model": selected,
            "selection": {"ppo_iteration": 0, "criterion": "strict mixed sampled"}})

    monkeypatch.setattr(subject, "train", train)
    monkeypatch.setattr(subject, "run_child", finalize)
    monkeypatch.setattr(chain.signal, "signal", lambda *args: None)
    subject.run()
    trained = [event for event in events if event[0] == "train"]
    assert [event[1] for event in trained] == ["v4", "v5", "v6"]
    assert trained[1][2] == "runs/v4/best.zip"
    assert trained[2][2] == "runs/v5/best.zip"
    if top3:
        assert [event for event in events if event[0] == "ranking"] == [
            ("ranking", "v5", 4), ("ranking", "v6", 5)]
    assert subject.state["status"] == "completed"
    assert subject.state["budget"]["used"] == subject.manifest["budget"]
    assert subject.state["budget"]["remaining"] == {
        "ppo_iterations": 0, "samples": 0, "adam_updates": 0}


def test_retention_restarts_after_partial_cleanup(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    folder = tmp_path / "experiments" / "v4"
    folder.mkdir(parents=True)
    models = tmp_path / "runs" / "v4"
    models.mkdir(parents=True)
    best, removed, remaining = [models / name for name in ("best.zip", "starting.zip", "final.zip")]
    for path in (best, removed, remaining):
        path.write_bytes(path.name.encode())
    selected = {"path": str(best), "sha256": chain.digest(best)}
    chain.write_json(folder / "retention.json", {"status": "planned", "kept_model": selected,
        "removed_models": [{"path": str(p.relative_to(tmp_path)), "sha256": chain.digest(p)}
                           for p in (removed, remaining)]})
    chain.write_json(folder / "resolved-config.json", {"name": "v4"})
    chain.write_json(folder / "experiment.json", {})
    chain.write_json(folder / "progress.json", {})
    removed.unlink()  # Simulate a crash after the first unlink.
    chain.complete_retention(folder)
    chain.complete_retention(folder)
    assert best.exists() and not remaining.exists()
    assert chain.read_json(folder / "retention.json")["status"] == "completed"
    assert chain.read_json(folder / "experiment.json")["status"] == "completed"


def test_binding_intent_survives_crash_before_config_write(tmp_path, monkeypatch):
    source = tmp_path / "v4-best.zip"
    checkpoint(source)
    folder = tmp_path / "v5"
    folder.mkdir()
    cfg = config()
    cfg.update(initial_model=None, initial_model_sha256=None)
    chain.write_json(folder / "config.json", cfg)
    chain.write_json(folder / "experiment.json", {})
    original_hash = chain.digest(folder / "config.json")
    previous = {"status": "completed", "model_version": "v4",
                "retained_model": chain.inspect_checkpoint(source),
                "selection": {"ppo_iteration": 0}}
    writer = chain.write_json

    def fail_config(path, value):
        if path.name == "config.json":
            raise OSError("injected crash after durable intent")
        writer(path, value)

    monkeypatch.setattr(chain, "write_json", fail_config)
    with pytest.raises(OSError, match="injected"):
        chain.bind_source(tmp_path, folder, previous)
    assert chain.digest(folder / "config.json") == original_hash
    assert chain.read_json(folder / "source-binding.json")["status"] == "pending"
    assert chain.read_json(folder / "source-binding.json")["previous_config_sha256"] == original_hash
    monkeypatch.setattr(chain, "write_json", writer)
    bound = chain.bind_source(tmp_path, folder, previous)
    assert bound["initial_model"] == "v4-best.zip"
    assert chain.read_json(folder / "source-binding.json")["status"] == "completed"


def test_controller_accepts_pending_source_intent_after_pool_binding(tmp_path, monkeypatch):
    subject = controller(tmp_path)
    folder = tmp_path / "v7"
    folder.mkdir()
    chain.write_json(folder / "config.json", {"config_status": "pool_bound"})
    intermediate_hash = chain.digest(folder / "config.json")
    chain.write_json(folder / "source-binding.json", {"status": "pending",
        "previous_config_sha256": intermediate_hash, "config_sha256": "1" * 64})
    subject.manifest["stages"] = [{"version": "v7", "experiment_dir": "v7",
                                   "prepared_config_sha256": "0" * 64}]
    monkeypatch.setattr(chain.signal, "signal", lambda *args: None)

    def reached_validation(config):
        raise RuntimeError("passed intermediate pool hash guard")

    monkeypatch.setattr(chain, "verify_config", reached_validation)
    with pytest.raises(RuntimeError, match="passed intermediate pool hash guard"):
        subject.run()


def test_repair100_fills_missing_validation_and_metrics_without_optimization(tmp_path, monkeypatch):
    from taverngym.learning import experiment, ppo, tracking

    cfg = config()
    models = tmp_path / "runs" / cfg["name"]
    models.mkdir(parents=True)
    saved = models / "latest_model.zip"
    checkpoint(saved, iteration=100, adam=7200)
    cfg["best_model_filename"] = "best.zip"
    chain.write_json(tmp_path / "resolved-config.json", cfg)
    chain.write_json(tmp_path / "progress.json", {"status": "interrupted"})
    for iteration in (0, 50):
        chain.write_json(tmp_path / f"evaluation-{iteration:04d}.json", evaluation(4))
    with (tmp_path / "metrics.jsonl").open("w") as stream:
        for iteration in range(1, 100):
            stream.write(json.dumps({"iteration": iteration,
                                    "train/optimizer_steps": iteration * 16}) + "\n")
    journal = tmp_path / "wandb-metrics.jsonl"
    journal.touch()

    class Model:
        last_rollout_seconds = 1.0
        last_train_seconds = 2.0

        def save(self, path):
            Path(path).write_bytes(saved.read_bytes())

    class Tracker:
        def __init__(self, *args):
            pass

        def log(self, values, optimizer_step):
            with journal.open("a") as stream:
                stream.write(json.dumps({**values, "optimizer_step": optimizer_step}) + "\n")

        def finish(self):
            pass

    calls = []

    def evaluate(model, games, seed, *args):
        calls.append((games, seed))
        result = evaluation(3, games)
        result["seed"] = seed
        for row in result["conditions"].values():
            row["seed"] = seed
        return result

    monkeypatch.setattr(ppo.EntityPPO, "load", lambda *a, **k: Model())
    monkeypatch.setattr(experiment, "load_frozen_opponents", lambda *a: None)
    monkeypatch.setattr(experiment, "evaluate_suite", evaluate)
    monkeypatch.setattr(tracking, "WandbTracker", Tracker)
    chain.repair_boundary(tmp_path, tmp_path, "cpu")
    assert calls == [(64, 40000003), (128, 50000003)]
    assert chain.read_json(tmp_path / "progress.json")["status"] == "completed"
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 100 and rows[-1]["recovered_boundary"]
    assert rows[-1]["train/optimizer_steps_this_iteration"] == 16
    assert chain.inspect_checkpoint(models / "final_model.zip")["adam_total_steps"] == 7200
    assert chain.read_json(tmp_path / "boundary-repair.json")["adam_updates_added"] == 0
