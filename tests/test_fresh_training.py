"""The fresh lineage preserves legacy runs and freezes selection before held-out tests."""

import hashlib
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from stable_baselines3.common.logger import configure

from taverngym.learning import experiment, tracking_v2
from taverngym.learning.metric_schema_v2 import SCHEMA_PATH, map_metric_payload_v2


def test_fresh_lineage_uses_sealed_v2_without_claiming_a_historical_version(tmp_path, monkeypatch):
    configurations = []
    monkeypatch.setattr(tracking_v2.WandbTracker, "__init__",
                        lambda self, folder, config, **kwargs: configurations.append(config))
    config = {"lineage": "eric-3702-fresh-v1", "model_version": "eric-3702-fresh-v1",
              "metric_schema_version": 2,
              "metric_schema_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()}
    assert isinstance(tracking_v2.tracker_for_config(tmp_path, config), tracking_v2.WandbTrackerV2)
    assert configurations[0]["model_version"] == "eric-3702-fresh-v1"
    for changed in ({"model_version": "v7"}, {"metric_schema_sha256": "0" * 64},
                    {"metric_schema_version": 1}):
        with pytest.raises(ValueError):
            tracking_v2.WandbTrackerV2(tmp_path, {**config, **changed})


def test_v0_evaluation_has_exactly_two_modes_and_sampled_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "configure_precision", lambda *args: None)

    def evaluate(model, games, seed, **kwargs):
        calls.append((games, seed, kwargs))
        return {"mean_placement": 2.0 if kwargs["deterministic"] else 4.0, "games": games}

    monkeypatch.setattr(experiment, "evaluate", evaluate)
    config = {"evaluation_modes": ["sampled", "greedy"], "frozen_opponents": 0}
    result = experiment.evaluate_suite(object(), 64, 20000101, config)
    assert len(calls) == 2
    assert all(games == 64 and seed == 20000101 and kwargs["opponent_policy"] is None
               for games, seed, kwargs in calls)
    assert [kwargs["deterministic"] for _, _, kwargs in calls] == [False, True]
    assert result["mean_placement"] == 4.0
    assert result["selection_condition"] == "mixed_sampled"
    with pytest.raises(ValueError, match="requires the frozen opponent"):
        experiment.evaluate_suite(object(), 64, 20000101, {"evaluation_matrix": True})


def test_heldout_loads_selected_best_without_touching_final(tmp_path, monkeypatch):
    path = tmp_path / "starting-best.zip"
    path.write_bytes(b"selected zero-step checkpoint")
    final = SimpleNamespace(num_timesteps=26214400)
    selected = SimpleNamespace(num_timesteps=0, optimizer_step_count=lambda: 0)
    loads, evaluations = [], []

    def load(checkpoint, **kwargs):
        loads.append((checkpoint, kwargs))
        return selected

    def evaluate(model, games, seed, config, opponent):
        evaluations.append((model, games, seed))
        return {"conditions": {"mixed_sampled": {"mean_placement": 5.},
                               "mixed_greedy": {"mean_placement": 6.}}}

    monkeypatch.setattr(experiment.EntityPPO, "load", load)
    monkeypatch.setattr(experiment, "evaluate_suite", evaluate)
    config = {"final_evaluation_target": "best", "final_evaluation_games": 128,
              "final_evaluation_seed": 30000101}
    result = experiment.evaluate_heldout(final, config, path, "cuda")
    assert loads[0][0] == path and loads[0][1]["n_steps"] == 2
    assert evaluations == [(selected, 128, 30000101)]
    assert final.num_timesteps == 26214400
    assert result["checkpoint_timesteps"] == result["checkpoint_optimizer_steps"] == 0
    assert map_metric_payload_v2(experiment.heldout_metric_payload(result, "best")) == {
        "test/core_best_sampled_mean_placement": 5.,
        "test/core_best_greedy_mean_placement": 6.,
    }


@pytest.mark.parametrize("first,expected", [(False, ["latest_model.zip"]),
                                            (True, ["latest_model.zip", "iteration-0001.zip"])])
def test_first_boundary_checkpoint_does_not_change_periodic_frequency(tmp_path, monkeypatch,
                                                                     first, expected):
    saved = []
    monkeypatch.setattr(experiment, "save_model", lambda model, path: saved.append(path.name))
    config = {"decisions_per_iteration": 262144, "iterations": 100,
              "checkpoint_every_iterations": 25, "checkpoint_first_iteration": first,
              "evaluation_every_iterations": 25, "evaluation_games": 0,
              "train_per_optimizer_step": False, "tracking_backend": "none"}
    model = SimpleNamespace(num_timesteps=262144, last_rollout_seconds=2., last_train_seconds=1.,
                            _rollout_places=[], optimizer_step_count=lambda: 32,
                            logger=configure(format_strings=[]))
    recorder = experiment.Recorder(config, tmp_path, tmp_path)
    recorder.model = model
    recorder.after_update(model)
    assert saved == expected
    assert recorder.completed == 1 and model.optimizer_step_count() == 32


def test_nonfinite_optimization_summary_never_replaces_the_clean_boundary(tmp_path, monkeypatch):
    saved = []
    monkeypatch.setattr(experiment, "save_model", lambda *args: saved.append(args))
    model = SimpleNamespace(num_timesteps=262144, last_rollout_seconds=2., last_train_seconds=1.,
                            _rollout_places=[], logger=configure(format_strings=[]))
    model.logger.record("train/approx_kl", float("inf"))
    recorder = experiment.Recorder({"decisions_per_iteration": 262144}, tmp_path, tmp_path)
    with pytest.raises(FloatingPointError, match="Non-finite"):
        recorder.after_update(model)
    assert saved == []
