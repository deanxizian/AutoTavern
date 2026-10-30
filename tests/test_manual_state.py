"""Offline human-entry validation is separate from model and live-client readiness."""

import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import sys
import uuid

import pytest

from tools.manual_state import check_transition, lookup, template, validate_entry
from tools.manual_advice import render_or_block


NOW = datetime(2026, 10, 30, 1, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def entry():
    state = template()
    state.update(observed_at=NOW.isoformat(), client_build="37.0.255790", confirmed=True,
                 phase="recruit", turn=3, hero={"card_id": "TB_BaconShop_HERO_01"},
                 resources={"health": 30, "armor": 10, "gold": 5, "tavern_tier": 2},
                 zones={"board": [], "hand": [], "shop": [{"card_id": "AT_121",
                     "attack": 9, "health": 12, "golden": False, "frozen": False,
                     "cost": {"amount": 2, "currency": "health"}}]},
                 hero_powers=[{"card_id": "TB_BaconShop_HP_001",
                               "cost": {"amount": 0, "currency": "gold"}}],
                 visible_effects=[], pending={"kind": "none", "options": [], "targets": []},
                 legal_actions={"confirmed": True, "visible_labels": ["购买酒馆第 1 位（2 生命值）", "冻结"]})
    return state


def codes(result):
    return {row["code"] for row in result["issues"]}


def test_complete_human_record_is_never_a_model_input_or_recommendation():
    value = entry()
    before = copy.deepcopy(value)
    result = validate_entry(value, now=NOW)
    assert result["status"] == "record_complete", result
    assert not result["model_ready"] and not result["live_ready"]
    assert "legal_action_mapping_unvalidated" in result["model_blockers"]
    assert not {"instruction", "mask", "me", "zones"} & result.keys()
    assert render_or_block(value, 0)["reason"] == "client_adapter_unvalidated"
    assert value == before


def test_template_keeps_unknown_fields_unknown_and_reports_paths():
    value = template()
    assert value["observed_at"] is None and value["zones"]["board"] is None
    result = validate_entry(value, now=NOW)
    assert result["status"] == "needs_input"
    assert {"unknown_card", "missing_legal_actions", "visible_state_unconfirmed"} <= codes(result)
    assert any(i["path"] == "zones.board" for i in result["issues"])


@pytest.mark.parametrize("delta,status", [(31, "record_stale"), (-10, "record_future")])
def test_capture_age_cannot_be_refreshed_by_checking_again(delta, status):
    value = entry()
    value["observed_at"] = (NOW - timedelta(seconds=delta)).isoformat()
    assert status in codes(validate_entry(value, now=NOW))


@pytest.mark.parametrize("time", [None, "bad", "2026-10-30T01:00:00", {}, 1])
def test_unknown_or_timezone_free_observation_time_is_blocked(time):
    value = entry()
    value["observed_at"] = time
    assert "record_unknown" in codes(validate_entry(value, now=NOW))


@pytest.mark.parametrize("mutate,code", [
    (lambda s: s["zones"]["shop"][0].update(card_id="UNKNOWN"), "unknown_card"),
    (lambda s: s["zones"]["shop"][0].update(attack=None), "missing_number"),
    (lambda s: s["zones"]["shop"][0].update(health=True), "missing_number"),
    (lambda s: s["zones"]["shop"][0].update(golden=True), "golden_id_mismatch"),
    (lambda s: s["zones"]["shop"][0].update(frozen=None), "unknown_frozen"),
    (lambda s: s["zones"]["shop"][0]["cost"].update(currency=None), "unknown_currency"),
    (lambda s: s["zones"]["shop"][0].update(card_id="TB_BaconShop_HERO_01"), "card_kind_mismatch"),
    (lambda s: s.update(legal_actions=None), "missing_legal_actions"),
    (lambda s: s["legal_actions"].update(visible_labels=[]), "missing_visible_labels"),
    (lambda s: s.update(opponents=[{"hand": "PRIVATE_OPPONENT"}]), "unsupported_fields"),
    (lambda s: s["zones"]["shop"][0].update(attack=float("nan")), "invalid_json_values"),
])
def test_missing_or_inconsistent_information_is_not_filled_from_catalog(mutate, code):
    value = entry()
    mutate(value)
    result = validate_entry(value, now=NOW)
    assert code in codes(result) and result["status"] == "needs_input", result
    assert "PRIVATE_OPPONENT" not in json.dumps(result)


def test_capture_digest_invalidates_changed_order_even_for_same_card_ids():
    value = entry()
    value["zones"]["shop"].append(copy.deepcopy(value["zones"]["shop"][0]))
    value["zones"]["shop"][1]["attack"] = 20
    prior = validate_entry(value, now=NOW)
    value["zones"]["shop"].reverse()
    changed = validate_entry(value, now=NOW, expected_state_id=prior["state_id"])
    assert "state_changed" in codes(changed)


def test_duplicate_entities_and_invalid_targets_are_blocked():
    value = entry()
    value["zones"]["shop"][0]["entity_id"] = 40
    value["zones"]["hand"] = copy.deepcopy(value["zones"]["shop"])
    value["pending"] = {"kind": "target", "options": [], "targets": [{"zone": "board", "position": 1}]}
    assert {"duplicate_entity", "unresolved_target"} <= codes(validate_entry(value, now=NOW))
    value["zones"]["hand"] = []
    value["pending"]["targets"][0]["zone"] = "shop"
    assert validate_entry(value, now=NOW)["status"] == "record_complete"


def test_choose_one_requires_visible_mode_labels_and_does_not_render_them():
    value = entry()
    option = {"card_id": "AT_121", "attack": 4, "health": 4, "golden": False}
    value["pending"] = {"kind": "choose_one", "options": [option], "targets": []}
    assert "unresolved_choose_one" in codes(validate_entry(value, now=NOW))
    option["label"] = "人工抄录的实际模式文字"
    value["pending"]["options"].append({**option, "label": "另一个实际模式文字"})
    result = validate_entry(value, now=NOW)
    assert result["status"] == "record_complete" and not result["live_ready"]


def test_hero_selection_does_not_invent_a_selected_hero_or_resources():
    value = entry()
    value.update(phase="hero_select", hero=None,
                 resources=dict.fromkeys(value["resources"]), hero_powers=[],
                 pending={"kind": "hero", "options": [{"card_id": "TB_BaconShop_HERO_01"}], "targets": []})
    value["zones"] = {zone: [] for zone in value["zones"]}
    assert validate_entry(value, now=NOW)["status"] == "record_complete"


def test_new_capture_required_after_manual_action_and_never_claims_execution():
    before = entry()
    after = copy.deepcopy(before)
    assert "capture_reused" in codes(check_transition(before, after, now=NOW))
    after["capture_id"] = uuid.uuid4().hex
    after["observed_at"] = (NOW + timedelta(seconds=1)).isoformat()
    assert "state_change_not_observed" in codes(check_transition(before, after, now=NOW))
    after["resources"]["gold"] = 2
    result = check_transition(before, after, now=NOW)
    assert result["status"] == "record_complete"
    assert result["transition"] == {"changed_fields": ["resources"], "execution_verified": False}
    after["observed_at"] = before["observed_at"]
    assert "capture_time_not_advanced" in codes(check_transition(before, after, now=NOW))


def test_historical_before_record_can_be_old_but_must_be_structurally_complete():
    before = entry()
    before["observed_at"] = (NOW - timedelta(minutes=5)).isoformat()
    after = entry()
    after["resources"]["gold"] = 2
    assert check_transition(before, after, now=NOW)["status"] == "record_complete"
    before["zones"]["shop"] = None
    assert "invalid_previous_record" in codes(check_transition(before, after, now=NOW))


@pytest.mark.parametrize("value", [None, [], "private text", 1])
def test_malformed_entry_or_transition_never_crashes_or_echoes_input(value):
    result = check_transition(entry(), value, now=NOW)
    assert result["status"] == "needs_input" and "private text" not in json.dumps(result)
    assert "wrong_schema" in codes(result)


def test_catalog_lookup_names_exact_id_and_never_claims_current_patch_support():
    result = lookup("人气选手")
    assert result["matches"][0]["id"] == "AT_121"
    assert not result["live_support_verified"]
    assert lookup("UNKNOWN_CARD")["matches"] == []


def test_cli_template_preserves_existing_work_and_check_has_actionable_exit(tmp_path):
    path = tmp_path / "entry.json"
    def run(*args):
        return subprocess.run([sys.executable, str(ROOT / "tools/manual_state.py"), *map(str, args)],
                              cwd=tmp_path, capture_output=True, text=True)
    assert run("template", "--out", path).returncode == 0
    original = path.read_bytes()
    assert path.stat().st_mode & 0o777 == 0o600
    second = run("template", "--out", path)
    assert second.returncode == 2 and json.loads(second.stdout)["reason"] == "output_exists"
    assert path.read_bytes() == original
    checked = run("check", path)
    assert checked.returncode == 2 and json.loads(checked.stdout)["status"] == "needs_input"
    assert run("check", path, "--out", path).returncode == 2
    assert path.read_bytes() == original
