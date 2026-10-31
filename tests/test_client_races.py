"""Race acquisition boundaries: incomplete memory, game changes and hung helpers."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time

import pytest

from tools.client_races import RaceReader, validate_read
from tools.window_session import WindowSession

SESSION = "Hearthstone_2026_10_31_12_00_00"
SOURCE = {"path": f"/logs/{SESSION}/Power.log", "device": 1, "inode": 2, "game_start_line": 1}
RACES = ["BEAST", "DEMON", "MECHANICAL", "MURLOC", "QUILBOAR"]


def response(key="game-a", handle=31, **changes):
    return {"schema": "autotavern.races.v1", "game_key": key, "status": "read", "pid": 100,
            "game_handle": handle, "log_session": SESSION, "game_type": 23,
            "race_ids": [20, 15, 17, 14, 43], **changes}


def wait_for(call, predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = call()
        if predicate(value):
            return value
        time.sleep(.005)
    raise AssertionError(f"Timed out: {value}")


@pytest.mark.parametrize("changes,expected", [
    ({"race_ids": [20, 15, 17, 14]}, "invalid_memory_response"),
    ({"race_ids": [20, 15, 17, 14, 999]}, "invalid_memory_response"),
    ({"race_ids": [20, 15, 17, 14, 14]}, "invalid_memory_response"),
    ({"race_ids": [20, 15, 17, 14, True]}, "invalid_memory_response"),
    ({"race_ids": [20, 15, 17, 14, 43, 0]}, "invalid_memory_response"),
    ({"game_key": "previous-game"}, "invalid_memory_response"),
    ({"log_session": "previous-client"}, "session_mismatch"),
    ({"game_handle": 0}, "invalid_memory_response"),
    ({"pid": -1}, "invalid_memory_response"),
    ({"game_type": 37}, "not_battlegrounds"),
    ({"status": "memory_access_denied", "mach_error": 5}, "memory_access_denied"),
    ({"status": "ready"}, "invalid_memory_response"),
    ({"status": []}, "invalid_memory_response"),
])
def test_memory_must_establish_exactly_five_known_races_for_this_solo_game(changes, expected):
    assert validate_read(response(**changes), "game-a", SESSION) == (expected, None)


def test_all_supported_races_include_aberration_without_dropping_new_ids():
    assert validate_read(response(), "game-a", SESSION) == ("ready", RACES)
    assert validate_read(response(race_ids=[11, 18, 24, 92, 126]), "game-a", SESSION) == (
        "ready", ["ABERRATION", "DRAGON", "ELEMENTAL", "NAGA", "UNDEAD"])
    assert validate_read([], "game-a", SESSION)[0] == "invalid_memory_response"


def test_retries_empty_reads_confirms_twice_then_stops_polling_and_redacts_diagnostics(tmp_path):
    calls = []
    def read(key):
        calls.append(key)
        return response(key, race_ids=[] if len(calls) == 1 else [20, 15, 17, 14, 43],
                        unrelated_secret="DO_NOT_WRITE_THIS")
    path = tmp_path / "races.jsonl"
    reader = RaceReader("unused", read=read, diagnostic_file=path, retry_scale=.01)
    try:
        sample = lambda: reader.sample("game-a", SOURCE, enabled=True)
        result = wait_for(sample, lambda r: r["status"] == "ready")
        assert result["races"] == RACES and result["attempt"] == 3
        time.sleep(.04)
        assert len(calls) == 3
        assert "DO_NOT_WRITE_THIS" not in path.read_text()
        assert path.stat().st_mode & 0o777 == 0o600
        assert "confirming" in path.read_text()
    finally:
        reader.close()


def test_delayed_old_game_result_cannot_fill_new_game():
    entered, release = threading.Event(), threading.Event()
    def read(key):
        if key == "game-a":
            entered.set()
            assert release.wait(2)
        return response(key, handle=31 if key == "game-a" else 32)
    reader = RaceReader("unused", read=read, retry_scale=.01)
    try:
        reader.sample("game-a", SOURCE, enabled=True)
        assert entered.wait(2)
        start = time.monotonic()
        sample = lambda: reader.sample("game-b", SOURCE, enabled=True)
        assert sample()["status"] == "reading"
        assert time.monotonic() - start < .1
        release.set()
        result = wait_for(sample, lambda r: r["status"] == "ready")
        assert result["game_key"] == "game-b" and result["game_handle"] == 32
    finally:
        release.set()
        reader.close()


def test_same_memory_handle_cannot_be_reused_for_another_log_game():
    handle = [31]
    reader = RaceReader("unused", read=lambda k: response(k, handle=handle[0]), retry_scale=.01)
    try:
        wait_for(lambda: reader.sample("game-a", SOURCE, enabled=True), lambda r: r["status"] == "ready")
        sample = lambda: reader.sample("game-b", SOURCE, enabled=True)
        assert wait_for(sample, lambda r: r["status"] == "stale_memory_game").get("races") is None
        handle[0] = 32
        assert wait_for(sample, lambda r: r["status"] == "ready")["game_handle"] == 32
    finally:
        reader.close()


def test_previous_game_handle_remains_bound_after_companion_restart():
    reader = RaceReader("unused", read=lambda k: response(k), retry_scale=.01)
    try:
        reader.restore_bindings({"game-a": {"source": "hearthmirror", "evidence": {
            "pid": 100, "log_session": SESSION, "game_handle": 31}}})
        sample = lambda: reader.sample("game-b", SOURCE, enabled=True)
        assert wait_for(sample, lambda r: r["status"] == "stale_memory_game").get("races") is None
    finally:
        reader.close()


def test_finishing_game_while_memory_is_reading_discards_late_result():
    release = threading.Event()
    reader = RaceReader("unused", read=lambda k: (release.wait(1), response(k))[1], retry_scale=.01)
    try:
        reader.sample("game-a", SOURCE, enabled=True)
        time.sleep(.02)
        reader.sample("game-a", SOURCE, enabled=False)
        release.set()
        time.sleep(.03)
        assert reader.sample("game-a", SOURCE, enabled=False)["status"] == "idle"
    finally:
        release.set()
        reader.close()


def test_hung_helper_is_killed_without_blocking_state_poll(tmp_path):
    helper = tmp_path / "hung-reader"
    helper.write_text("#!/bin/sh\nexec sleep 10\n")
    helper.chmod(0o700)
    reader = RaceReader(helper, timeout=.08)
    try:
        sample = lambda: reader.sample("game-a", SOURCE, enabled=True)
        result = wait_for(sample, lambda r: r["status"] == "helper_timeout")
        assert "races" not in result
        assert reader.process is None
    finally:
        reader.close()


def test_missing_helper_reports_an_actionable_error_without_raising(tmp_path):
    reader = RaceReader(tmp_path / "missing")
    try:
        sample = lambda: reader.sample("game-a", SOURCE, enabled=True)
        assert wait_for(sample, lambda r: r["status"] == "helper_missing")["detail"]
    finally:
        reader.close()


class Feed:
    def __init__(self):
        self.source = deepcopy(SOURCE)
        self.complete = False
    def sample(self, force=False):
        return {"running": True, "snapshot": {"source": self.source, "entities": [],
            "status": "incomplete_snapshot", "game_status": "complete" if self.complete else "running",
            "freshness": {"status": "recent"}}}


def test_window_automatically_persists_memory_races_and_passes_them_to_model(tmp_path):
    feed = Feed()
    reader = RaceReader("unused", read=lambda k: response(k), retry_scale=.01)
    class Model:
        seen = None
        def sample(self, snapshot, enabled):
            self.seen = snapshot
    model = Model()
    try:
        receipt = tmp_path / "session/feedback.jsonl"
        session = WindowSession(feed, receipt_file=receipt, race_reader=reader, model_bridge=model)
        state = wait_for(session.state, lambda r: bool(r["lobby_races"]))
        assert state["lobby_races"] == RACES and not state["needs_lobby_races"]
        assert state["lobby_races_source"] == "hearthmirror"
        assert model.seen["lobby_races"] == RACES
        stored = json.loads(session.races_file.read_text())[state["game_key"]]
        assert stored["source"] == "hearthmirror" and stored["evidence"]["game_handle"] == 31
        restarted = WindowSession(feed, receipt_file=receipt)
        assert restarted.state()["lobby_races"] == RACES
        feed.source["game_start_line"] = 500
        assert session.state()["needs_lobby_races"] and session.state()["lobby_races"] is None
        feed.complete = True
        assert not session.state()["needs_lobby_races"]
    finally:
        reader.close()
