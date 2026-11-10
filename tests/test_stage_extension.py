"""Conditional budget and development ranking never use heldout or version val."""

import json

import pytest

pytest.importorskip("torch")

from taverngym.learning import stage_chain as chain
from taverngym.learning import stage_extension as extension


def gate_fixture(root, rate):
    results = {}
    stages = []
    for version in ("v4", "v5", "v6"):
        folder = root / version
        folder.mkdir()
        chain.write_json(folder / "retention.json", {"status": "completed"})
        results[version] = {"status": "completed", "fresh_samples": 13107200,
                           "verification": {"finite_and_protocol_consistent": True},
                           "selection": {"ppo_iteration": 0}}
        stages.append({"version": version, "experiment_dir": version})
    chain.write_json(root / "v6/evaluation-0000.json", {
        "mean_placement": 3, "cycle_forfeit_rate": rate, "action_limit_forfeit_rate": 0})
    (root / "v6/evaluation-best-heldout.json").write_text("not read by gate")
    return {"stages": stages}, results


@pytest.mark.parametrize("rate,eligible", [(1, False), (.5, True)])
def test_gate_rejects_all_self_forfeits_even_when_placement_improved(tmp_path, rate, eligible):
    manifest, results = gate_fixture(tmp_path, rate)
    gate = extension.extension_gate(tmp_path, manifest, results)
    assert gate["eligible"] is eligible
    assert not gate["heldout_used_for_gate"] and not gate["used_for_model_selection"]


def test_gate_rejects_unresolved_native_or_numerical_problem(tmp_path):
    manifest, results = gate_fixture(tmp_path, .5)
    results["v4"]["verification"]["finite_and_protocol_consistent"] = False
    (tmp_path / "v5/native-action-failure.json").write_text("{}")
    gate = extension.extension_gate(tmp_path, manifest, results)
    assert not gate["eligible"]
    assert not gate["checks"]["v4_finite_and_consistent"]
    assert not gate["checks"]["v5_no_unresolved_native_failure"]


@pytest.mark.parametrize("value", [None, float("nan"), -.1, 1.1])
def test_gate_rejects_missing_nonfinite_or_invalid_forfeit_rate(tmp_path, value):
    manifest, results = gate_fixture(tmp_path, .5)
    path = tmp_path / "v6/evaluation-0000.json"
    selected = chain.read_json(path)
    if value is None:
        selected.pop("action_limit_forfeit_rate")
    else:
        selected["action_limit_forfeit_rate"] = value
    chain.write_json(path, selected)
    gate = extension.extension_gate(tmp_path, manifest, results)
    assert not gate["eligible"] and not gate["checks"]["v6_self_forfeit_metrics_valid"]


@pytest.mark.parametrize("cycle,limit,eligible", [(0, .5, True), (0, 1, False), (.1, 0, False)])
def test_soft_cycle_gate_checks_disabled_cycle_forfeit_and_action_budget_only(
        tmp_path, cycle, limit, eligible):
    manifest, results = gate_fixture(tmp_path, cycle)
    manifest["cycle_rule"] = "constant_penalty_no_cycle_forfeit"
    path = tmp_path / "v6/evaluation-0000.json"
    selected = chain.read_json(path)
    selected.update(action_limit_forfeit_rate=limit, cycle_return_fraction=1)
    chain.write_json(path, selected)
    gate = extension.extension_gate(tmp_path, manifest, results)
    assert gate["eligible"] is eligible
    assert "v6_selected_best_not_all_self_forfeits" not in gate["checks"]


@pytest.mark.parametrize("cycle,limit,eligible", [(0, 0, True), (.1, 0, False), (0, .1, False)])
def test_forced_end_gate_treats_zero_forfeits_as_integrity_not_model_improvement(
        tmp_path, cycle, limit, eligible):
    manifest, results = gate_fixture(tmp_path, cycle)
    manifest["cycle_rule"] = "constant_penalty_forced_end_no_forfeits"
    path = tmp_path / "v6/evaluation-0000.json"
    selected = chain.read_json(path)
    selected.update(action_limit_forfeit_rate=limit, cycle_return_fraction=1,
                    max_consecutive_cycle_returns=59)
    chain.write_json(path, selected)
    gate = extension.extension_gate(tmp_path, manifest, results)
    assert gate["eligible"] is eligible
    assert gate["checks"]["v6_disabled_early_termination_rates_are_zero"] is eligible
    assert not any("not_all" in key for key in gate["checks"])


def test_three_step0_bests_stop_conditional_spending_without_heldout(tmp_path):
    manifest, results = gate_fixture(tmp_path, 0)
    manifest.update(cycle_rule="constant_penalty_forced_end_no_forfeits",
                    stop_extension_after_three_step0_bests=True)
    gate = extension.extension_gate(tmp_path, manifest, results)
    assert not gate["eligible"] and not gate["checks"]["not_three_consecutive_stage_best_step0"]
    results["v5"]["selection"]["ppo_iteration"] = 50
    assert extension.extension_gate(tmp_path, manifest, results)["eligible"]


def test_conditional_budget_is_unavailable_until_primary_gate_passes(tmp_path, monkeypatch):
    manifest, results = gate_fixture(tmp_path, .5)
    manifest.update(instance_id="test", root=str(tmp_path),
        primary_budget={"ppo_iterations": 300, "samples": 39321600, "adam_updates": 4800},
        budget={"ppo_iterations": 600, "samples": 78643200, "adam_updates": 9600},
        conditional_stages=[{"version": version} for version in ("v7", "v8", "v9")])
    path = tmp_path / "manifest.json"
    chain.write_json(path, manifest)
    subject = chain.StageChain(path)
    assert subject.budget(0, {})["remaining"] == manifest["primary_budget"]
    stages = list(manifest["stages"])
    subject.maybe_extend(stages, 1, results)
    assert len(stages) == 3
    subject.maybe_extend(stages, 2, results)
    assert [stage["version"] for stage in stages] == [f"v{n}" for n in range(4, 10)]
    assert subject.state["conditional_extension_enabled"]
    assert subject.budget(3, {})["remaining"] == manifest["budget"]


@pytest.mark.parametrize("top3", [False, True])
@pytest.mark.parametrize("checkpoint_move_cost", [0, .0005])
def test_ranking_uses_own_protocol_reference_and_older_tie(
        tmp_path, monkeypatch, top3, checkpoint_move_cost):
    from taverngym.learning import experiment, ppo, precision

    candidates = []
    current = {"operation_cost": 0, "cycle_cost": .0002, "cycle_detection": "all_actions",
               "cycle_forfeit_repetitions": 3, "turn_action_forfeit_limit": 60}
    for number in range(1, 6):
        path = tmp_path / f"v{number}.zip"
        path.write_text(str(number))
        candidates.append({"model_version": f"v{number}", "name": f"bot_v{number}",
                           "model": path.name, "sha256": chain.digest(path), "count": 1,
                           "reward_shaping": {**current, "move_cost": 0} if number >= 4 else None})
    chain.write_json(tmp_path / "ranking-input.json", {"protected_version": None if top3 else "v6",
        "pool_selection": "top3" if top3 else "remove_weakest", "candidate_count": 5,
        "candidates": candidates, "protocol": {"games": 64, "seed": 60000007,
        "reference_pool": [], "version_order": [f"v{n}" for n in range(1, 10)]}})
    monkeypatch.setattr(extension, "inspect_checkpoint", lambda path: {
        "reward_shaping": ({**current, **({"move_cost": checkpoint_move_cost}
                                         if checkpoint_move_cost else {})}
                           if int(path.stem[1:]) >= 4 else None)})
    monkeypatch.setattr(experiment, "load_frozen_opponents", lambda *args: None)
    monkeypatch.setattr(ppo.EntityPPO, "load", lambda path, **kwargs: path.stem)
    monkeypatch.setattr(precision, "configure_precision", lambda *args: None)
    calls = []

    def evaluate(model, games, seed, **kwargs):
        calls.append((model, games, seed, kwargs["reward_shaping"]))
        placement = 5 if model in ("v1", "v4") else 4
        if top3 and model == "v5":
            placement = 6  # The latest generation is allowed to miss the pool.
        return {"mean_placement": placement, "placements": [placement] * games,
                "games": games, "seed": seed}

    monkeypatch.setattr(experiment, "evaluate", evaluate)
    if checkpoint_move_cost:
        with pytest.raises(ValueError, match="actual generation protocol"):
            extension.rank_candidates(tmp_path, tmp_path, "cpu")
        assert [call[0] for call in calls] == ["v1", "v2", "v3"]
        return
    extension.rank_candidates(tmp_path, tmp_path, "cpu")
    result = chain.read_json(tmp_path / "ranking-results.json")
    if top3:
        assert result["selected_versions"] == ["v3", "v2", "v4"]
        assert result["protected_version"] is None
        config = {"initial_model": None, "opponent_pool": [], "expected_opponent_counts": None}
        chain.write_json(tmp_path / "config.json", config)
        bound = extension.bind_ranked_pool(tmp_path, tmp_path, {"model_version": "v5"})
        assert [spec["model_version"] for spec in bound["opponent_pool"]] == ["v3", "v2", "v4"]
        assert all(spec["count"] == 2 for spec in bound["opponent_pool"])
        assert bound["initial_model"] is None
    else:
        assert result["removed_version"] == "v1"
    assert len(calls) == 5 and all(games == 64 and seed == 60000007
                                 for _, games, seed, _ in calls)
    assert all(protocol is None for _, _, _, protocol in calls[:3])
    assert all(protocol == current for _, _, _, protocol in calls[3:])
    extension.rank_candidates(tmp_path, tmp_path, "cpu")
    assert len(calls) == 5
    # Simulate termination after three sealed candidates: reuse those exact
    # records and evaluate only the two that do not yet have a sealed result.
    (tmp_path / "ranking-v4.json").unlink()
    (tmp_path / "ranking-v5.json").unlink()
    extension.rank_candidates(tmp_path, tmp_path, "cpu")
    assert [call[0] for call in calls[5:]] == ["v4", "v5"]
    bad = chain.read_json(tmp_path / "ranking-v1.json")
    bad["result"]["seed"] = 1
    chain.write_json(tmp_path / "ranking-v1.json", bad)
    with pytest.raises(ValueError, match="Sealed ranking result"):
        extension.rank_candidates(tmp_path, tmp_path, "cpu")
    assert len(calls) == 7


def test_top3_registry_keeps_unselected_history_and_adds_every_completed_best(tmp_path):
    historical = {f"v{n}": {"model_version": f"v{n}", "name": f"bot_v{n}",
                             "model": f"v{n}.zip", "sha256": str(n), "count": 2,
                             "reward_shaping": None} for n in range(1, 4)}
    manifest = {"root": str(tmp_path), "pool_selection": "top3", "historical_models": historical,
                "ranking_protocol": {"version_order": [f"v{n}" for n in range(1, 10)]}}
    results = {f"v{n}": {"model_version": f"v{n}",
        "retained_model": {"path": str(tmp_path / f"v{n}-best.zip"), "sha256": str(n)},
        "reward_shaping": {"cycle_cost": .01}, "config_path": str(tmp_path / f"v{n}/config.json"),
        "opponent_pool": [historical["v1"], historical["v2"]]} for n in (4, 5)}
    first = extension.ranking_input(manifest, results["v4"], {"v4": results["v4"]})
    assert first["candidate_count"] == 4
    second = extension.ranking_input(manifest, results["v5"], results)
    assert second["candidate_count"] == 5
    assert [spec["model_version"] for spec in second["candidates"]] == [f"v{n}" for n in range(1, 6)]
    assert any(spec["model_version"] == "v3" for spec in second["candidates"])
    assert second["protected_version"] is None


def lifecycle_fixture(root, previous_version="v5"):
    versions = {f"v{n}": {"status": "retired" if n in (4, 6) else "active",
                           "canonical_version": f"v{3 if n == 4 else 5 if n == 6 else n}"}
                for n in range(7)}
    chain.write_json(root / "registry.json", {"schema_version": 1, "versions": versions})
    historical = {}
    for n in (1, 2, 3):
        path = root / f"v{n}.zip"
        path.write_text(str(n))
        historical[f"v{n}"] = {"model_version": f"v{n}", "name": f"bot_v{n}",
            "model": path.name, "sha256": chain.digest(path), "count": 2, "reward_shaping": None}
    path = root / f"{previous_version}-best.zip"
    path.write_text(previous_version)
    previous = {"model_version": previous_version, "retained_model": {
        "path": str(path), "sha256": chain.digest(path)}, "reward_shaping": None,
        "config_path": str(root / f"{previous_version}/config.json")}
    manifest = {"root": str(root), "pool_selection": "top3", "historical_models": historical,
        "model_version_registry": "registry.json", "excluded_opponent_versions": ["v4", "v6"],
        "ranking_protocol": {"version_order": [f"v{n}" for n in range(1, 10)],
            "reference_pool": list(historical.values()), "games": 64, "seed": 60000007}}
    return manifest, previous


def test_lifecycle_registry_excludes_retired_without_loading_aliases(tmp_path):
    manifest, previous = lifecycle_fixture(tmp_path)
    inputs = extension.ranking_input(manifest, previous)
    assert [x["model_version"] for x in inputs["candidates"]] == ["v1", "v2", "v3", "v5"]
    assert inputs["learning_source_version"] == "v5" and inputs["candidate_count"] == 4
    assert inputs["model_version_registry_sha256"] == chain.digest(tmp_path / "registry.json")
    assert set(inputs["excluded_opponent_versions"]) == {"v4", "v6"}
    manifest["ranking_protocol"]["reference_pool"] = [{"model_version": "v4"}]
    with pytest.raises(ValueError, match="Retired or unregistered"):
        extension.ranking_input(manifest, previous)


@pytest.mark.parametrize("problem", ["missing_active", "active_alias", "spec_alias"])
def test_lifecycle_rejects_missing_candidates_or_alias_versions(tmp_path, problem):
    manifest, previous = lifecycle_fixture(tmp_path)
    if problem == "missing_active":
        del manifest["historical_models"]["v2"]
    elif problem == "spec_alias":
        manifest["historical_models"]["v2"]["model_version"] = "v1"
    else:
        document = chain.read_json(tmp_path / "registry.json")
        document["versions"]["v2"]["canonical_version"] = "v1"
        chain.write_json(tmp_path / "registry.json", document)
    with pytest.raises(ValueError, match="available historical|canonical|alias"):
        extension.ranking_input(manifest, previous)


def test_lifecycle_three_candidate_ranking_uses_sealed_snapshot_not_global_file(tmp_path, monkeypatch):
    from taverngym.learning import experiment, ppo, precision

    manifest, previous = lifecycle_fixture(tmp_path, "v3")
    inputs = extension.ranking_input(manifest, previous)
    assert inputs["candidate_count"] == 3
    chain.write_json(tmp_path / "ranking-input.json", inputs)
    (tmp_path / "registry.json").unlink()  # Sealed input must be self-contained.
    monkeypatch.setattr(extension, "inspect_checkpoint", lambda path: {"reward_shaping": None})
    monkeypatch.setattr(experiment, "load_frozen_opponents", lambda *a: None)
    monkeypatch.setattr(ppo.EntityPPO, "load", lambda path, **k: path.stem)
    monkeypatch.setattr(precision, "configure_precision", lambda *a: None)
    calls = []

    def evaluate(model, games, seed, **kwargs):
        calls.append(model)
        return {"mean_placement": 4, "placements": [4] * games, "games": games, "seed": seed}

    monkeypatch.setattr(experiment, "evaluate", evaluate)
    extension.rank_candidates(tmp_path, tmp_path, "cpu")
    extension.rank_candidates(tmp_path, tmp_path, "cpu")
    assert len(calls) == 3  # Existing sealed candidates are reused, never rerun.
    chain.write_json(tmp_path / "config.json", {"initial_model": "v5-best.zip"})
    bound = extension.bind_ranked_pool(tmp_path, tmp_path, previous)
    assert bound["initial_model"] == "v5-best.zip"
    assert [x["model_version"] for x in bound["opponent_pool"]] == ["v3", "v2", "v1"]
    assert all(x["count"] == 2 for x in bound["opponent_pool"])
    ranking = chain.read_json(tmp_path / "ranking-results.json")
    ranking["selected_versions"][0] = "v4"
    chain.write_json(tmp_path / "ranking-results.json", ranking)
    with pytest.raises(ValueError, match="Retired version cannot occupy"):
        extension.bind_ranked_pool(tmp_path, tmp_path, previous)


@pytest.mark.parametrize("registry_change", ["deleted", "retired_after_sealing"])
def test_controller_resume_uses_sealed_lifecycle_input_without_global_registry(
        tmp_path, monkeypatch, registry_change):
    manifest, previous = lifecycle_fixture(tmp_path)

    def inspect(path):
        return {"path": str(path.resolve()), "sha256": chain.digest(path),
                "ppo_iteration": 100, "adam_total_steps": 7200}

    previous.update(status="completed", fresh_samples=13107200, actual_adam_updates=1600,
                    selection={"ppo_iteration": 100})
    previous["retained_model"] = inspect(tmp_path / "v5-best.zip")
    old, folder = tmp_path / "v5", tmp_path / "v7"
    old.mkdir()
    folder.mkdir()
    chain.write_json(old / "results.json", previous)
    chain.write_json(folder / "experiment.json", {})
    chain.write_json(folder / "config.json", {"name": "v7", "initial_model": None,
        "opponent_pool": [], "derived_budget": {}, "reward_shaping": {"move_cost": .0005}})
    inputs = extension.ranking_input(manifest, previous, {"v5": previous})
    chain.write_json(folder / "ranking-input.json", inputs)
    input_hash = chain.digest(folder / "ranking-input.json")
    chain.write_json(folder / "ranking-v1.json", {"sealed": "preserve existing candidate"})
    partial_hash = chain.digest(folder / "ranking-v1.json")
    if registry_change == "deleted":
        (tmp_path / "registry.json").unlink()
    else:
        document = chain.read_json(tmp_path / "registry.json")
        document["versions"]["v3"] = {"status": "retired", "canonical_version": "v2"}
        chain.write_json(tmp_path / "registry.json", document)
    manifest.update(instance_id="sealed-resume-test", budget={
        "ppo_iterations": 200, "samples": 26214400, "adam_updates": 3200}, stages=[
            {"version": "v5", "experiment_dir": "v5"},
            {"version": "v7", "experiment_dir": "v7", "rank_opponents": True,
             "prepared_config_sha256": chain.digest(folder / "config.json")}])
    chain.write_json(tmp_path / "manifest.json", manifest)
    subject = chain.StageChain(tmp_path / "manifest.json", resume=True)
    events = []

    def worker(command, stage, index, phase):
        events.append(phase)
        if phase == "ranking":
            assert chain.digest(folder / "ranking-input.json") == input_hash
            assert chain.digest(folder / "ranking-v1.json") == partial_hash
            chain.write_json(folder / "ranking-results.json", {"status": "completed",
                "pool_selection": "top3", "input_sha256": input_hash,
                "selected_versions": ["v3", "v2", "v1"]})
        else:
            assert phase == "finalizing"
            result = {**previous, "model_version": "v7", "retained_model": inspect(
                tmp_path / "runs/v7/best.zip")}
            chain.write_json(folder / "results.json", result)

    def train(stage, index, folder, cfg, resume=False):
        events.append("mock_training")
        assert cfg["initial_model"] == "v5-best.zip"
        model = tmp_path / "runs/v7/best.zip"
        model.parent.mkdir(parents=True)
        model.write_text("mock result; no training performed")

    monkeypatch.setattr(extension, "ranking_input", lambda *a, **k:
                        pytest.fail("A sealed ranking must never reread the global registry"))
    monkeypatch.setattr(chain, "inspect_checkpoint", inspect)
    monkeypatch.setattr(chain, "complete_retention", lambda folder: None)
    monkeypatch.setattr(chain, "verify_config", lambda config: None)
    monkeypatch.setattr(chain.signal, "signal", lambda *a: None)
    monkeypatch.setattr(subject, "run_child", worker)
    monkeypatch.setattr(subject, "train", train)
    subject.run()
    assert events == ["ranking", "mock_training", "finalizing"]
    assert subject.state["status"] == "completed"
    assert chain.digest(folder / "ranking-input.json") == input_hash
    assert chain.digest(folder / "ranking-v1.json") == partial_hash


@pytest.mark.parametrize("tamper", ["source", "candidate", "candidate_protocol", "protocol"])
def test_sealed_lifecycle_recovery_still_rejects_source_or_protocol_changes(tmp_path, tamper):
    manifest, previous = lifecycle_fixture(tmp_path)
    inputs = json.loads(json.dumps(extension.ranking_input(manifest, previous)))
    (tmp_path / "registry.json").unlink()
    if tamper == "source":
        (tmp_path / "v5-best.zip").write_text("changed source")
    elif tamper == "candidate":
        (tmp_path / "v1.zip").write_text("changed candidate")
    elif tamper == "candidate_protocol":
        inputs["candidates"][-1]["reward_shaping"] = {"move_cost": .0005}
    else:
        inputs["protocol"]["games"] = 65
    with pytest.raises(ValueError, match="source changed|checkpoint changed|candidates changed|protocol changed"):
        chain.validate_sealed_lifecycle_ranking(tmp_path, manifest, previous, {}, inputs)


def test_resuming_completed_primary_stages_still_enters_conditional_sequence(tmp_path, monkeypatch):
    stages = []
    for number in range(4, 10):
        version = f"v{number}"
        folder = tmp_path / version
        folder.mkdir()
        model = folder / "best.zip"
        model.write_text(version)
        chain.write_json(folder / "results.json", {"status": "completed",
            "model_version": version, "fresh_samples": 13107200, "actual_adam_updates": 1600,
            "retained_model": {"path": str(model), "sha256": chain.digest(model)}, "selection": {}})
        stages.append({"version": version, "experiment_dir": version, "conditional": number >= 7})
    manifest = {"root": str(tmp_path), "instance_id": "resumed-primary", "stages": stages[:3],
        "conditional_stages": stages[3:],
        "primary_budget": {"ppo_iterations": 300, "samples": 39321600, "adam_updates": 4800},
        "budget": {"ppo_iterations": 600, "samples": 78643200, "adam_updates": 9600}}
    path = tmp_path / "manifest.json"
    chain.write_json(path, manifest)
    subject = chain.StageChain(path, resume=True)
    monkeypatch.setattr(chain, "complete_retention", lambda folder: None)
    monkeypatch.setattr(chain.signal, "signal", lambda *args: None)
    monkeypatch.setattr(extension, "extension_gate", lambda *args: {"eligible": True})
    monkeypatch.setattr(subject, "train", lambda *args: pytest.fail("unexpected retraining"))
    subject.run()
    assert [stage["version"] for stage in subject.state["completed_stages"]] == [
        f"v{n}" for n in range(4, 10)]
    assert subject.state["conditional_extension_enabled"] and subject.state["status"] == "completed"
