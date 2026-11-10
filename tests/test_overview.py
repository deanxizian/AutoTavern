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


def test_fresh_overview_keeps_zero_step_best_and_counts_only_saved_complete_updates(tmp_path,
                                                                                  monkeypatch):
    from stable_baselines3.common import save_util

    root = tmp_path
    experiments = root / 'experiments'
    folder = experiments / 'fresh'
    folder.mkdir(parents=True)
    (experiments / 'experiment-overview.json').write_text(json.dumps({'stages': {'v9': {'keep': 1}}}))
    (experiments / 'EXPERIMENT_OVERVIEW.md').write_text('Historical results stay here.\n')
    # PPO2's rollout is full, but PPO2's update has not completed or been saved.
    (folder / 'progress.json').write_text(json.dumps({
        'completed_iterations': 1, 'sampled_decisions': 524288, 'optimizer_steps': 33,
    }))
    run = root / 'runs/fresh'
    run.mkdir(parents=True)
    (run / 'zero-best.zip').write_bytes(b'zero-step best')
    (run / 'latest_model.zip').write_bytes(b'complete PPO1')

    def load(path, **kwargs):
        if path.name == 'zero-best.zip':
            return {'num_timesteps': 0}, {'policy.optimizer': {'state': {}}}, None
        return {'num_timesteps': 262144, '_n_updates': 1}, {}, None

    monkeypatch.setattr(overview, '__file__', str(root / 'python/taverngym/learning/overview.py'))
    monkeypatch.setattr(save_util, 'load_from_zip_file', load)
    config = {'lineage': 'eric-3702-fresh-v1', 'metric_schema_version': 2,
              'metric_schema_sha256': 'schema', 'instance_id': 'fresh', 'rules_version': '37.0.2',
              'name': 'fresh', 'best_model_filename': 'zero-best.zip', 'iterations': 100,
              'initial_model_sha256': 'initial', 'decisions_per_iteration': 262144,
              'n_epochs': 1, 'evaluation_games': 64}
    overview.update_stage_overview(folder, config)
    document = json.loads((experiments / 'experiment-overview.json').read_text())
    assert document['stages'] == {'v9': {'keep': 1}}
    fresh = document['fresh_lineages']['eric-3702-fresh-v1']
    assert fresh['completed_ppo'] == 1 and fresh['completed_boundary_samples'] == 262144
    assert fresh['fresh_samples'] == 524288
    assert fresh['best']['ppo'] == 0 and fresh['best']['adam'] == 0
    assert fresh['roster'] == '7 V0'
