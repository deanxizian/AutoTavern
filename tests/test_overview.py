"""Completed training and an active retained model are distinct states."""

import json

import pytest

pytest.importorskip("stable_baselines3")

from taverngym.learning import overview, stage_chain


def test_completed_overview_keeps_lifecycle_separate_and_excludes_game_records(tmp_path, monkeypatch):
    root = tmp_path
    experiments = root / "experiments"
    folder = experiments / "v8"
    folder.mkdir(parents=True)
    (experiments / "experiment-overview.json").write_text(json.dumps({"stages": {}}))
    (experiments / "EXPERIMENT_OVERVIEW.md").write_text("Historical results stay here.\n")
    (folder / "progress.json").write_text(json.dumps({
        "completed_iterations": 100, "sampled_decisions": 13107200, "optimizer_steps": 1600,
    }))
    (folder / "evaluation-0050.json").write_text(json.dumps({"mean_placement": 4.109375}))
    (folder / "results.json").write_text(json.dumps({"heldout": {
        "best": {"conditions": {"mixed_sampled": {
            "games": 128, "mean_placement": 4.3046875,
            "placements": [4] * 128, "episodes": [{"placement": 4}],
        }}},
    }}))
    (root / "registry.json").write_text(json.dumps({"versions": {
        "v8": {"status": "active", "canonical_version": "v8"},
    }}))
    model = root / "runs" / "v8" / "best_model.zip"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"checkpoint stub")
    monkeypatch.setattr(overview, "__file__", str(root / "python/taverngym/learning/overview.py"))
    monkeypatch.setattr(stage_chain, "inspect_checkpoint", lambda _: {
        "ppo_iteration": 50, "adam_total_steps": 9600, "sha256": "model-sha",
    })
    config = {
        "metric_schema_version": 2, "metric_schema_sha256": "schema-sha",
        "model_version": "v8", "model_version_registry": "registry.json",
        "instance_id": "test-v89", "name": "v8", "best_model_filename": model.name,
        "iterations": 100, "initial_model_version": "v7", "initial_model_sha256": "source-sha",
        "opponent_pool": [{"count": 3, "model_version": "v5"},
                          {"count": 4, "model_version": "v7"}], "evaluation_games": 64,
    }
    overview.update_stage_overview(folder, config, completed=True)
    stage = json.loads((experiments / "experiment-overview.json").read_text())["stages"]["v8"]
    assert stage["status"] == "completed"
    assert stage["lifecycle_status"] == "active"
    assert stage["completed_ppo"] == 100
    assert stage["best_ppo"] == 50
    assert stage["heldout"]["best"]["mixed_sampled"] == {
        "games": 128, "mean_placement": 4.3046875,
    }
    markdown = (experiments / "EXPERIMENT_OVERVIEW.md").read_text()
    assert markdown.startswith("Historical results stay here.")
    assert "训练与收尾 completed；模型生命周期 active" in markdown
