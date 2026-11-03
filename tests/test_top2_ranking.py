"""V9 development selection: canonical candidates, seven frozen seats, full concurrency."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from taverngym.learning import parallel_ranking as parallel
from taverngym.learning import stage_chain as chain
from taverngym.learning import stage_extension as extension


def fixture(root, *, unchanged=False):
    historical = {}
    versions = {}
    for number in range(1, 9):
        version = f"v{number}"
        retired = number in (4, 6) or (number == 8 and unchanged)
        canonical = "v3" if number == 4 else "v5" if number == 6 else "v7" if retired else version
        versions[version] = {"status": "retired" if retired else "active",
                             "canonical_version": canonical}
        path = root / f"{version}.zip"
        path.write_text(version)
        historical[version] = {"model_version": version, "name": f"bot_{version}",
            "model": path.name, "sha256": chain.digest(path), "count": 2, "reward_shaping": None}
    chain.write_json(root / "registry.json", {"schema_version": 1, "versions": versions})
    previous = {"model_version": "v8", "retained_model": {
        "path": str(root / "v8.zip"), "sha256": historical["v8"]["sha256"]},
        "reward_shaping": None, "config_path": "v8/config.json"}
    manifest = {"root": str(root), "pool_selection": "top2", "historical_models": historical,
        "model_version_registry": "registry.json", "excluded_opponent_versions": ["v4", "v6"],
        "ranking_protocol": {"version_order": list(historical), "games": 64, "seed": 60000007,
            "precision": "bf16", "learner_mode": "stochastic", "parallel_all_candidates": True,
            "reference_pool": [dict(historical["v5"], count=3),
                               dict(historical["v7"], count=4)]}}
    return manifest, previous


@pytest.mark.parametrize("unchanged", [False, True])
def test_top2_input_excludes_aliases_but_keeps_physical_learning_source(tmp_path, unchanged):
    manifest, previous = fixture(tmp_path, unchanged=unchanged)
    inputs = extension.ranking_input(manifest, previous)
    assert [s["model_version"] for s in inputs["candidates"]] == (
        ["v1", "v2", "v3", "v5", "v7"] + ([] if unchanged else ["v8"]))
    assert inputs["learning_source_version"] == "v8"
    assert inputs["model_version_registry_sha256"] == chain.digest(tmp_path / "registry.json")
    assert sum(s["count"] for s in inputs["protocol"]["reference_pool"]) == 7
    (tmp_path / "registry.json").unlink()
    extension.validate_lifecycle_input(inputs)  # Fully sealed, no mutable registry access.


@pytest.mark.parametrize("field,value", [("parallel_all_candidates", False),
    ("games", 32), ("seed", 1), ("precision", "fp32"), ("learner_mode", "greedy")])
def test_top2_requires_locked_parallel_protocol(tmp_path, field, value):
    manifest, previous = fixture(tmp_path)
    manifest["ranking_protocol"][field] = value
    with pytest.raises(ValueError, match="parallel BF16 protocol"):
        extension.ranking_input(manifest, previous)


def test_top2_parallel_ranking_ties_bind_four_three_and_reuse_sealed_results(tmp_path, monkeypatch):
    manifest, previous = fixture(tmp_path)
    inputs = extension.ranking_input(manifest, previous)
    chain.write_json(tmp_path / "ranking-input.json", inputs)
    chain.write_json(tmp_path / "config.json", {"initial_model": "v8.zip"})
    monkeypatch.setattr(extension, "inspect_checkpoint", lambda path: {"reward_shaping": None})
    calls = []

    def evaluate(root, folder, actual_inputs, checkpoints, device):
        calls.append(actual_inputs)
        assert len(checkpoints) == 6 and device == "cpu"
        return {s["model_version"]: {"games": 64, "seed": 60000007,
            "mean_placement": 3 if s["model_version"] in ("v3", "v7") else 4,
            "placements": [3 if s["model_version"] in ("v3", "v7") else 4] * 64}
            for s in actual_inputs["candidates"]}

    monkeypatch.setattr(parallel, "parallel_evaluate", evaluate)
    extension.rank_candidates(tmp_path, tmp_path, "cpu")
    extension.rank_candidates(tmp_path, tmp_path, "cpu")
    assert len(calls) == 1
    ranking = chain.read_json(tmp_path / "ranking-results.json")
    assert ranking["selected_versions"] == ["v7", "v3"]
    assert ranking["heldout_used"] is False
    bound = extension.bind_ranked_pool(tmp_path, tmp_path, previous)
    assert [(s["model_version"], s["count"]) for s in bound["opponent_pool"]] == [
        ("v7", 4), ("v3", 3)]
    assert bound["initial_model"] == "v8.zip"
    ranking["selected_versions"].reverse()
    chain.write_json(tmp_path / "ranking-results.json", ranking)
    with pytest.raises(ValueError, match="sealed development ranking"):
        extension.bind_ranked_pool(tmp_path, tmp_path, previous)


def test_top2_native_scheduler_all_eight_seats_no_v0_and_cross_candidate_batch(tmp_path, monkeypatch):
    seen = []

    class Policy:
        def predict(self, obs, deterministic):
            assert deterministic is False
            seen.append(len(obs["action_mask"]))
            return np.array([0 if row[0] else np.flatnonzero(row)[0]
                             for row in obs["action_mask"]]), None

    specs = [{"model_version": v, "model": v, "sha256": v, "count": 2,
              "name": v, "reward_shaping": None} for v in ("v1", "v2")]
    checkpoint = lambda path: {"sha256": path.name, "reward_shaping": None,
                               "policy_sha256": path.name}
    monkeypatch.setattr(parallel, "digest", lambda path: path.name)
    monkeypatch.setattr(parallel, "inspect_checkpoint", checkpoint)
    monkeypatch.setattr(parallel, "load_frozen_policy", lambda *a, **k: Policy())
    result = parallel.parallel_evaluate(tmp_path, tmp_path, {
        "pool_selection": "top2", "candidates": specs,
        "protocol": {"games": 64, "seed": 60000007,
                     "reference_pool": [dict(specs[0], count=3), dict(specs[1], count=4)]}},
        {}, "cpu")
    games = [json.loads(line) for line in (tmp_path / "ranking-games.jsonl").read_text().splitlines()]
    assert len(games) == 128
    assert all(len(g["players"]) == 8 and g["v0_placements"] == [] for g in games)
    assert all(sorted(p["placement"] for p in g["players"]) == list(range(1, 9)) for g in games)
    scheduler = chain.read_json(tmp_path / "ranking-scheduler.json")
    assert scheduler["actual_initial_concurrency"] == scheduler["completed_games"] == 128
    assert scheduler["controlled_seats_per_game"] == 8 and scheduler["v0_seats_per_game"] == 0
    assert scheduler["replacement_games"] == 0 and scheduler["rejected_actions"] == 0
    assert all(any(shards == 2 for _, shards in batches)
               for batches in scheduler["batch_distribution"].values())
    assert max(seen) > 64 * 4
    assert all(len(row["placements"]) == 64 for row in result.values())


def test_top2_duplicate_policy_candidates_rejected_before_creating_lobbies(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "digest", lambda _: "same")
    monkeypatch.setattr(parallel, "inspect_checkpoint", lambda _: {
        "sha256": "same", "reward_shaping": None, "policy_sha256": "duplicate"})
    specs = [{"model_version": v, "model": v, "sha256": "same", "reward_shaping": None}
             for v in ("v1", "v2")]
    with pytest.raises(ValueError, match="retired aliases"):
        parallel.parallel_evaluate(tmp_path, tmp_path, {"pool_selection": "top2",
            "candidates": specs, "protocol": {"games": 64, "seed": 60000007,
                "reference_pool": [dict(specs[0], count=7)]}}, {}, "cpu")
    assert not (tmp_path / "ranking-games.jsonl").exists()
