"""The prepared future schema is complete, unambiguous and value preserving."""

import json

import pytest

from taverngym.learning.metric_schema_v2 import (
    SCHEMA_PATH, map_metric_payload_v2, metric_name_v2,
)


@pytest.mark.parametrize("old,new", [
    ("train/first_rate", "train_core/first_rate"),
    ("val/mean_placement", "val_core/mean_placement"),
    ("val/greedy_top4_rate", "val_core/greedy_top4_rate"),
    ("train/placement_reward", "train_reward/placement"),
    ("val/hero_selection_bonus", "val_reward/hero_selection"),
    ("val/greedy_action_reward", "val_reward/greedy_operation"),
    ("train/total_reward", "train_reward/total"),
    ("train/cycle_penalty", "train_reward/cycle"),
    ("train/cycle_returns", "train/operation_cycle_returns"),
    ("val/move_actions", "val/operation_move_actions"),
    ("train/first_turn_minion_bonus_count", "train/operation_first_turn_minion_milestone_count"),
    ("train/hero_selection_bonus_count", "train/operation_hero_selection_preference_hits"),
    ("train/loss", "train/optimization_adam_loss"),
    ("train/ppo_loss", "train/optimization_ppo_loss"),
    ("train/adam_total_steps", "train/progress_adam_total_steps"),
    ("train/ppo_optimizer_steps", "train/progress_adam_steps"),
    ("val/ppo_iteration", "val/progress_ppo_iteration"),
    ("system/rollout_seconds", "train/performance_rollout_seconds"),
    ("system/actor_bot_v8_batch_mean", "train/performance_actor_bot_v8_batch_mean"),
    ("test/best_mixed_greedy_mean_placement", "test/core_best_greedy_mean_placement"),
    ("test/initial_mixed_sampled_total_reward", "test/reward_initial_sampled_total"),
    ("optimizer_step", "optimizer_step"),
])
def test_only_core_and_reward_split_into_flat_sections(old, new):
    assert metric_name_v2(old) == new
    assert metric_name_v2(new) == new
    assert new.count("/") <= 1


def test_all_episode_fields_cover_train_val_greedy_and_all_six_heldout_scopes():
    schema = json.loads(SCHEMA_PATH.read_text())
    assert len(schema["base_metrics"]) == 55
    names = []
    for base in schema["base_metrics"]:
        names.extend([f"train/{base}", f"val/{base}", f"val/greedy_{base}"])
        names.extend(f"test/{stage}_mixed_{mode}_{base}"
                     for stage in ("initial", "best", "final")
                     for mode in ("sampled", "greedy"))
    mapped = [metric_name_v2(name) for name in names]
    assert len(mapped) == len(set(mapped)) == 55 * 9
    assert all(metric_name_v2(name) == name for name in mapped)
    assert {name.split("/")[0] for name in mapped} == set(schema["panel_sections"])
    assert set(schema["panel_sections"]) == {"train_core", "train_reward", "train",
                                             "val_core", "val_reward", "val", "test"}
    assert set(schema["stages"]) == {"train", "val", "test"}


@pytest.mark.parametrize("alias,normal", [
    ("rollout/mean_placement", "train/mean_placement"),
    ("baseline/first_rate", "val/first_rate"),
    ("eval/greedy_total_reward", "val/greedy_total_reward"),
    ("ppo_summary/approx_kl", "train/ppo_approx_kl"),
    ("time/update_seconds", "system/update_seconds"),
    ("progress/ppo_iterations", "train/ppo_iteration"),
    ("progress/sampled_decisions", "train/sampled_decisions"),
])
def test_internal_aliases_use_the_same_display_name(alias, normal):
    assert metric_name_v2(alias) == metric_name_v2(normal)


def test_payload_preserves_values_axis_and_consistent_aliases():
    values = {"val/mean_placement": 3.359375, "train/total_reward": -.42,
              "train/optimizer_steps": 400, "train/ppo_optimizer_steps": 400,
              "optimizer_step": 400}
    assert map_metric_payload_v2(values) == {
        "val_core/mean_placement": 3.359375, "train_reward/total": -.42,
        "train/progress_adam_steps": 400, "optimizer_step": 400,
    }
    with pytest.raises(ValueError, match="Conflicting"):
        map_metric_payload_v2({"train/optimizer_steps": 400, "train/ppo_optimizer_steps": 399})


@pytest.mark.parametrize("name", ["random_field", "train/unknown_new_metric", "core/bogus",
                                 "test/best_mystery_mean_placement"])
def test_unknown_fields_cannot_silently_create_unclassified_charts(name):
    with pytest.raises(ValueError, match="Unclassified"):
        metric_name_v2(name)


def test_this_is_a_v8_preparation_not_live_v7_integration():
    schema = json.loads(SCHEMA_PATH.read_text())
    assert schema["effective_from_model_version"] == "v8"
    assert schema["status"] == "prepared_not_integrated_with_active_v7"
    assert "_rate" not in schema["base_metrics"]["mean_placement"]["name"]


def test_wandb_system_is_reserved_for_sdk_hardware_telemetry():
    schema = json.loads(SCHEMA_PATH.read_text())
    assert "system" not in schema["families"]
    assert "performance" in schema["families"]
    for metric in schema["performance_metrics"]:
        target = metric_name_v2("system/" + metric)
        assert target.startswith("train/performance_")
        assert metric_name_v2(target) == target
    with pytest.raises(ValueError, match="Unclassified"):
        metric_name_v2("system/cpu")


def test_future_tracker_preserves_values_and_writes_each_mode_once(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    import sys
    from types import SimpleNamespace

    from taverngym.learning.tracking_v2 import WandbTrackerV2

    sent, configurations = [], []
    run = SimpleNamespace(entity="eric_eric_", project="AutoTavern", id="mock-v8", url="mock",
        define_metric=lambda *a, **k: None, log=sent.append, finish=lambda **k: None)

    def init(**kwargs):
        configurations.append(kwargs["config"])
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init, Settings=lambda **k: k))
    config = {"model_version": "v8", "metric_schema_version": 2,
              "metric_schema_sha256": __import__("hashlib").sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
              "tracking": {"mode": "offline"}}
    tracker = WandbTrackerV2(tmp_path, config)
    try:
        tracker.log({"rollout/mean_placement": 3.5, "rollout/action_reward": -.2,
                     "rollout/cycle_returns": 20, "system/rollout_seconds": 2.5,
                     "not_a_scalar": "ignored"}, 16)
        tracker.log({"val/mean_placement": 3.25, "val/greedy_mean_placement": 3.75}, 400)
        tracker.log({"test/best_mixed_sampled_mean_placement": 3.125}, 400)
        with pytest.raises(ValueError, match="actual Adam axis"):
            tracker.log({"optimizer_step": 399}, 400)
    finally:
        tracker.finish()
    rows = [json.loads(line)
            for line in (tmp_path / "wandb-metrics.jsonl").read_text().splitlines()]
    assert rows == sent == [
        {"train_core/mean_placement": 3.5, "train_reward/operation": -.2,
         "train/operation_cycle_returns": 20, "train/performance_rollout_seconds": 2.5,
         "optimizer_step": 16},
        {"val_core/mean_placement": 3.25, "val_core/greedy_mean_placement": 3.75,
         "optimizer_step": 400},
        {"test/core_best_sampled_mean_placement": 3.125, "optimizer_step": 400},
    ]
    assert configurations[0]["metric_schema_version"] == 2
    assert config["metric_schema_version"] == 2


@pytest.mark.parametrize("version", [None, "v1", "v7", "unknown"])
def test_future_tracker_refuses_to_change_a_v7_or_unspecified_run(tmp_path, version):
    pytest.importorskip("stable_baselines3")
    from taverngym.learning.tracking_v2 import WandbTrackerV2

    with pytest.raises(ValueError, match="V8\\+"):
        WandbTrackerV2(tmp_path, {"model_version": version})


def test_v8_output_boundary_resume_and_finalization_share_canonical_dedup(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    import hashlib
    import sys
    from types import SimpleNamespace
    from stable_baselines3.common.logger import configure
    from taverngym.learning.experiment import log_evaluation
    from taverngym.learning.rewards import REWARD_METRICS
    from taverngym.learning.tracking import WandbOutputFormat
    from taverngym.learning.tracking_v2 import tracker_for_config, journal_has_metric

    sent = []
    run = SimpleNamespace(entity="eric_eric_", project="AutoTavern", id="v8", url="mock",
        define_metric=lambda *a, **k: None, log=sent.append, finish=lambda **k: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **k: run,
                                                             Settings=lambda **k: k))
    cfg = {"model_version": "v8", "metric_schema_version": 2,
           "metric_schema_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
           "tracking": {"mode": "offline"}}
    for missing in ("metric_schema_version", "metric_schema_sha256"):
        with pytest.raises(ValueError):
            tracker_for_config(tmp_path, {k: v for k, v in cfg.items() if k != missing})
    tracker = tracker_for_config(tmp_path, cfg)
    episode = {**dict.fromkeys(REWARD_METRICS, 0.), "mean_placement": 3.5,
               "first_rate": .25, "top4_rate": .75, "cap_end_rate": .1}
    # The real experiment output adapter receives train, val, greedy, PPO and
    # performance together; the Adam callback logs directly through the tracker.
    tracker.log({"train/loss": 1., "train/learning_rate": 1e-5,
                 "train/optimizer_steps": 16}, 16)
    logger = configure(format_strings=[])
    logger.output_formats.append(WandbOutputFormat(tracker))
    for key, value in episode.items():
        logger.record("rollout/" + key, value)
    logger.record("ppo_summary/loss", 1.)
    logger.record("time/rollout_seconds", 2.)
    logger.record("progress/ppo_iterations", 1)
    log_evaluation(logger, "val", {"conditions": {
        "mixed_sampled": episode, "mixed_greedy": episode}})
    logger.dump(16)
    for checkpoint in ("initial", "best", "final"):
        tracker.log({f"test/{checkpoint}_mixed_{mode}_{key}": value
                     for mode in ("sampled", "greedy")
                     for key, value in episode.items()}, 16)
    tracker.finish()
    before = list(sent)
    resumed = tracker_for_config(tmp_path, cfg)
    for row in before:
        resumed.log(row, row["optimizer_step"])
    resumed.finish()
    assert sent == before
    rows = [json.loads(line) for line in (tmp_path / "wandb-metrics.jsonl").read_text().splitlines()]
    assert journal_has_metric(rows, "val/mean_placement", cfg, 16)
    assert journal_has_metric(rows, "test/best_mixed_sampled_mean_placement", cfg)
    assert {k.split("/")[0] for row in rows for k in row if "/" in k} == {
        "train_core", "train_reward", "train", "val_core", "val_reward", "val", "test"}
    for scope in ("train_reward", "val_reward"):
        assert all(any(f"{scope}/{metric}" in row for row in rows)
                   for metric in ("placement", "hero_selection", "operation", "total"))


def test_zero_adam_ppo_still_records_distinct_outcomes_and_replays_once(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    import hashlib
    import sys
    from types import SimpleNamespace
    from taverngym.learning.tracking_v2 import tracker_for_config

    sent = []
    run = SimpleNamespace(entity="eric_eric_", project="AutoTavern", id="zero-adam", url="mock",
        define_metric=lambda *a, **k: None, log=sent.append, finish=lambda **k: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **k: run,
                                                             Settings=lambda **k: k))
    cfg = {"model_version": "v8", "metric_schema_version": 2,
           "metric_schema_sha256": hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
           "tracking": {"mode": "offline"}}
    rows = [{"progress/ppo_iterations": iteration,
             "rollout/mean_placement": placement, "time/update_seconds": 1.}
            for iteration, placement in ((1, 4.), (2, 3.))]
    tracker = tracker_for_config(tmp_path, cfg)
    for row in rows:
        tracker.log(row, 16)
    tracker.log({"train/loss": 1.}, 16)
    tracker.finish()
    resumed = tracker_for_config(tmp_path, cfg)
    for row in rows:
        resumed.log(row, 16)
    resumed.log({"train/loss": 1.}, 16)
    resumed.finish()
    assert [r["train_core/mean_placement"] for r in sent
            if "train_core/mean_placement" in r] == [4., 3.]
    assert len(sent) == 3
