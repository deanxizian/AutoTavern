"""Memory identifies the client; positions and choices stay in the log stream."""
from copy import deepcopy
import json
import threading
import time

import pytest

from tools.client_actions import frame_identity
from tools.client_common import content_id
from tools.client_memory import PublicStateReader, merge_public_state, validate_public_read
from tools.window_session import WindowSession


SOURCE = {"path": "/logs/Hearthstone_current/Power.log", "device": 1, "inode": 2, "game_start_line": 1}
KEY = content_id([SOURCE[k] for k in ("path", "device", "inode", "game_start_line")])
EVIDENCE = {"pid": 100, "log_session": "Hearthstone_current", "game_handle": 31}


def response(key=KEY, **changes):
    return {"schema": "autotavern.memory.v1", "status": "public_read", "game_key": key, **EVIDENCE,
            "game_type": 23, "public_state": {"scene": {"mode": 4, "loaded": True, "transitioning": False},
            "spectator": False, "local_player_id": 3}, **changes}


def memory(**changes):
    return {**validate_public_read(response(), KEY, "Hearthstone_current"), "game_key": KEY,
            "stable": True, "identity_stable": True, **changes}


def snapshot():
    return {"source": deepcopy(SOURCE), "status": "snapshot", "game_status": "running", "client_phase": "recruit",
            "freshness": {"status": "recent"}, "model_snapshot_id": "log-id",
            "perspective": {"player_id": 11, "source": "hslog_inference_unvalidated_for_this_game"},
            "validation": {"issues": [{"code": "perspective_unconfirmed"}, {"code": "other"}]},
            "model_blockers": ["perspective_unconfirmed", "other"],
            "entities": [
                {"entity_id": 1, "type": "GAME", "tags": {"BACON_IN_COMBAT_PHASE": 0}},
                {"entity_id": 8, "type": "PLAYER", "controller": 3, "tags": {"PLAYER_ID": 3}},
                {"entity_id": 9, "type": "PLAYER", "controller": 11, "tags": {"PLAYER_ID": 11, "HERO_ENTITY": 56}},
                {"entity_id": 56, "type": "HERO", "controller": 11, "zone": "PLAY", "card_id": "TB_BaconShopBob"},
                {"entity_id": 101, "type": "MINION", "controller": 3, "zone": "PLAY", "position": 1, "card_id": "BG31_815"},
                {"entity_id": 201, "type": "MINION", "controller": 11, "zone": "PLAY", "position": 1, "card_id": "BG20_101"}]}


def wait_for(call, predicate, timeout=3):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        result = call()
        if predicate(result): return result
        time.sleep(.005)
    raise AssertionError(result)


def test_exact_memory_identity_corrects_inference_and_invalidates_old_model_frame_without_mutating_log():
    snap = snapshot(); before = deepcopy(snap)
    merged, scene, audit = merge_public_state(snap, memory(), EVIDENCE)
    assert snap == before
    assert merged["perspective"] == {"player_id": 3, "source": "hearthmirror"}
    assert frame_identity(merged) != frame_identity(snap)
    assert "perspective_unconfirmed" not in merged["model_blockers"]
    assert merged["validation"]["issues"] == [{"code": "other"}]
    assert scene == "对局界面" and not audit["blocks_actions"]
    assert merged["entities"] is snap["entities"]
    # Merely confirming the same player must not re-run inference or reset advice.
    inferred = {**merged, "perspective": {"player_id": 3, "source": "log"}}
    assert frame_identity(merged) == frame_identity(inferred)


@pytest.mark.parametrize("changes", [
    {"game_key": "previous-game"}, {"log_session": "previous-client"},
    {"game_handle": 30}, {"pid": 99}, {"status": "stale"}, {"identity_stable": False}, {"game_type": 37},
])
def test_stale_wrong_game_and_unconfirmed_memory_cannot_change_player_or_positions(changes):
    snap = snapshot()
    merged, _, _ = merge_public_state(snap, memory(**changes), EVIDENCE)
    assert merged == snap


def test_memory_needs_a_verified_game_binding_even_when_races_were_manually_filled():
    snap = snapshot()
    assert merge_public_state(snap, memory(), None)[0] == snap


@pytest.mark.parametrize("phase", ["recruit", "combat"])
def test_conflicting_legacy_memory_cards_never_change_log_positions_choices_or_readiness(phase):
    snap = snapshot(); snap["client_phase"] = phase
    snap["last_choices"] = {"entity_ids": [101, 201], "selection_sent_or_resolved": False}
    mem = memory()
    mem["public_state"].update(
        friendly_board=[{"entity_id": 101, "card_id": "BG31_815", "position": 7}],
        opposing_board=[], choices={"visible": True, "cards": ["BG20_101", "BG31_815"]},
        mulligan={"waiting": True, "cards": []})
    merged, _, audit = merge_public_state(snap, mem, EVIDENCE)
    assert merged["entities"] is snap["entities"] and merged["last_choices"] is snap["last_choices"]
    assert not audit["blocks_actions"]
    assert not {"board", "shop", "choices", "positions"} & audit.keys()


@pytest.mark.parametrize("changes", [
    {"scene": {"mode": 4, "loaded": True, "transitioning": True}},
    {"spectator": True}, {"local_player_id": 17},
])
def test_transition_spectating_or_absent_player_cannot_adopt_a_memory_perspective(changes):
    mem = memory(); mem["public_state"].update(changes)
    snap = snapshot(); merged, _, audit = merge_public_state(snap, mem, EVIDENCE)
    assert merged == snap
    if "local_player_id" not in changes: assert audit["blocks_actions"]


def test_legacy_card_payloads_are_discarded_at_the_transport_boundary():
    raw = response()
    raw["public_state"].update(friendly_board=[{"card_id": "BG31_815", "position": 1}],
                               opposing_board=[], choices={"visible": True, "cards": ["BG20_101"]},
                               mulligan={"waiting": True, "cards": []})
    value = validate_public_read(raw, KEY, "Hearthstone_current")
    assert set(value["public_state"]) == {"scene", "local_player_id", "spectator"}
    assert value["public_state"]["local_player_id"] == 3


def test_private_or_unknown_properties_never_reach_diagnostics(tmp_path):
    raw = response(); raw["secret"] = "DO_NOT_EXPORT"
    raw["public_state"].update(account="DO_NOT_EXPORT", opponent_hand=["DO_NOT_EXPORT"])
    path = tmp_path/"memory.jsonl"
    reader = PublicStateReader("unused", read=lambda k: raw, interval=.01, diagnostic_file=path)
    try:
        result = wait_for(lambda: reader.sample(KEY, SOURCE), lambda r: r.get("stable"))
        assert "DO_NOT_EXPORT" not in json.dumps(result)
        assert "DO_NOT_EXPORT" not in path.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
    finally: reader.close()


def test_legacy_board_animation_cannot_delay_identity_confirmation_or_trigger_diagnostics():
    count = [0]
    def read(key):
        count[0] += 1
        raw = response(key)
        raw["public_state"]["friendly_board"] = [{"entity_id": 100+count[0], "card_id": "BG31_815", "position": 1}]
        return raw
    reader = PublicStateReader("unused", read=read, interval=.01)
    try:
        result = wait_for(lambda: reader.sample(KEY, SOURCE), lambda r: r.get("identity_stable"))
        assert result["stable"]
        merged, _, _ = merge_public_state(snapshot(), result, EVIDENCE)
        assert merged["perspective"]["player_id"] == 3
    finally: reader.close()


def test_a_hung_refresh_expires_last_read_without_blocking_ui():
    release = threading.Event(); calls = [0]
    def read(key):
        calls[0] += 1
        if calls[0] > 2: release.wait(2)
        return response(key)
    reader = PublicStateReader("unused", read=read, interval=.005, max_age=.04)
    try:
        call = lambda: reader.sample(KEY, SOURCE)
        wait_for(call, lambda r: r.get("stable"))
        stale = wait_for(call, lambda r: r["status"] == "stale")
        assert "public_state" not in stale
        start = time.monotonic(); call()
        assert time.monotonic()-start < .05
    finally:
        release.set(); reader.close()


def test_late_old_game_results_are_discarded_after_context_change():
    entered, release = threading.Event(), threading.Event()
    def read(key):
        if key == KEY:
            entered.set(); release.wait(2)
        return response(key)
    reader = PublicStateReader("unused", read=read, interval=.005)
    try:
        reader.sample(KEY, SOURCE); assert entered.wait(2)
        assert reader.sample("new", SOURCE)["status"] == "reading"
        release.set()
        result = wait_for(lambda: reader.sample("new", SOURCE), lambda r: r.get("stable"))
        assert result["game_key"] == "new"
        assert reader.sample("new", SOURCE, enabled=False)["status"] == "idle"
    finally:
        release.set(); reader.close()


def test_hung_helper_is_replaced_and_last_state_is_not_reused(tmp_path):
    helper = tmp_path/"reader"
    helper.write_text("#!/bin/sh\nexec sleep 10\n"); helper.chmod(0o700)
    reader = PublicStateReader(helper, timeout=.04, interval=.005)
    try:
        value = wait_for(lambda: reader.sample(KEY, SOURCE), lambda r: r["status"] == "helper_timeout")
        assert "public_state" not in value and reader.transport.process is None
    finally: reader.close()


def test_window_consumes_memory_scene_and_player_and_withdraws_advice_outside_game():
    snap = snapshot()
    class Feed:
        def sample(self, force=False):
            return {"running": True, "scene": "旧日志界面", "snapshot": snap}
    class Memory:
        value = memory()
        def sample(self, *args, **kwargs): return self.value
    class Model:
        seen = None
        enabled = None
        def sample(self, snapshot, enabled):
            self.seen, self.enabled = snapshot, enabled
            return {"status": "waiting_for_game"}
    reader, model = Memory(), Model()
    session = WindowSession(Feed(), memory_reader=reader, model_bridge=model)
    session.race_records[KEY] = {"evidence": EVIDENCE, "races": ["BEAST", "ELEMENTAL", "MECHANICAL", "MURLOC", "QUILBOAR"]}
    state = session.state()
    assert state["client_scene"] == "对局界面" and state["client_scene_source"] == "hearthmirror"
    assert state["perspective"]["player_id"] == model.seen["perspective"]["player_id"] == 3
    assert model.enabled
    reader.value["public_state"]["scene"]["mode"] = 15
    state = session.state()
    assert state["client_scene"] == "酒馆战棋大厅" and not model.enabled
    assert not state["live_ready"] and state["model_status"]["title"] == "等待对局界面"
    # Helper failure falls back without making the log reader or model unusable.
    reader.value = {"status": "helper_timeout"}
    state = session.state()
    assert model.enabled and state["client_scene_source"] == "loading_screen_log"
