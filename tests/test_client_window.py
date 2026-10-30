"""The native companion only previews recorded actions, with bound user feedback."""

from copy import deepcopy
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tools.client_window import make_server
from tools.window_session import WindowError, WindowSession, latest_scene, rehearsal_frame


class FakeFeed:
    def sample(self, force=False):
        return {"running": True, "build": "37.0.255790", "snapshot": {
            "status": "missing_log", "model_ready": True, "live_ready": True,
            "source": {"path": "/fictional/Hearthstone_2026_10_30_09_11_28/Power.log"}}}


def test_scene_reads_only_completed_current_mode_not_a_requested_transition(tmp_path):
    path = tmp_path / "LoadingScreen.log"
    path.write_text("PrivateAccount currMode=HUB\nprevMode=HUB nextMode=PACKOPENING\n")
    assert latest_scene(path) == "主界面"
    path.write_text(path.read_text() + "currMode=PACKOPENING")
    assert latest_scene(path) == "主界面"
    path.write_text(path.read_text() + "\n")
    assert latest_scene(path) == "开包界面"


def test_missing_unknown_or_incomplete_scene_never_invents_a_game(tmp_path):
    path = tmp_path / "LoadingScreen.log"
    assert latest_scene(path) is None
    path.write_text("currMode=UNKNOWN\n")
    assert latest_scene(path) is None
    path.write_text("currMode=PACKOPENING")
    assert latest_scene(path) is None


def test_live_feed_cannot_enable_advice_with_ready_flags_or_a_mask():
    session = WindowSession(FakeFeed())
    state = session.state()
    assert state["mode"] == "live" and state["proposal"] is None
    assert not state["model_ready"] and not state["live_ready"]
    with pytest.raises(WindowError, match="实战建议"):
        session.publish_preview({"scope": "live", "view": {}, "action": 0, "live_ready": True})
    with pytest.raises(WindowError):
        session.next_frame()


def test_feedback_is_once_only_and_requires_a_new_frame_before_next_instruction(tmp_path):
    receipts = tmp_path / "feedback.jsonl"
    session = WindowSession(FakeFeed(), receipt_file=receipts)
    first = session.set_mode("rehearsal")
    proposal = first["proposal"]
    assert "购买" in proposal["instruction"]
    assert proposal["scope"] == "simulator_action_preview"
    after = session.feedback(proposal["id"], "executed")
    assert after["proposal"] is None and after["can_next_frame"]
    assert after["cards"] == first["cards"]  # Feedback itself is not a state transition.
    assert after["history"][0]["execution_verified"] is False
    with pytest.raises(WindowError) as error:
        session.feedback(proposal["id"], "executed")
    assert error.value.code == "stale_proposal"
    assert len(receipts.read_text().splitlines()) == 1
    assert receipts.stat().st_mode & 0o777 == 0o600
    next_state = session.next_frame()
    assert next_state["proposal"]["title"] == "打出随从"
    assert next_state["proposal"]["frame_id"] != proposal["frame_id"]


def test_decline_does_not_advance_rehearsal_or_report_execution():
    session = WindowSession(FakeFeed())
    proposal = session.set_mode("rehearsal")["proposal"]
    declined = session.feedback(proposal["id"], "not_executed")
    assert declined["history"][0]["outcome"] == "not_executed"
    reloaded = session.next_frame()["proposal"]
    assert reloaded["title"] == proposal["title"] and reloaded["id"] != proposal["id"]


def test_expiry_and_mode_changes_invalidate_old_buttons():
    clock = [100.]
    session = WindowSession(FakeFeed(), clock=lambda: clock[0], ttl=10)
    first = session.set_mode("rehearsal")["proposal"]
    clock[0] = 111
    assert session.state()["proposal"] is None
    with pytest.raises(WindowError):
        session.feedback(first["id"], "executed")
    second = session.next_frame()["proposal"]
    session.set_mode("live")
    with pytest.raises(WindowError):
        session.feedback(second["id"], "executed")
    assert session.state()["cards"] == []


def test_imported_preview_exports_no_private_opponents_and_does_not_mutate_view():
    view, action = rehearsal_frame(0)
    view["opponents"] = [{"hand": "PRIVATE_OPPONENT"}]
    before = deepcopy(view)
    session = WindowSession(FakeFeed())
    state = session.publish_preview({"scope": "simulator_action_preview", "view": view, "action": action})
    assert state["mode"] == "replay" and not state["live_ready"]
    assert "PRIVATE_OPPONENT" not in json.dumps(state)
    assert "PRIVATE_OPPONENT" not in json.dumps(session.frame)
    assert view == before


def test_invalid_import_keeps_previous_mode_and_proposal():
    session = WindowSession(FakeFeed())
    prior = session.set_mode("rehearsal")["proposal"]
    with pytest.raises(WindowError):
        session.publish_preview({"scope": "simulator_action_preview", "view": {"model_ready": False}, "action": 0})
    assert session.state()["proposal"]["id"] == prior["id"]
    assert session.state()["mode"] == "rehearsal"


@pytest.fixture
def service():
    session = WindowSession(FakeFeed())
    server = make_server(session, "test-local-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", session
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def request(service, path, data=None, *, authenticated=True, origin=None, host=None):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = "Bearer test-local-token"
    if origin:
        headers["Origin"] = origin
    if host:
        headers["Host"] = host
    req = Request(service[0] + path, data=json.dumps(data).encode() if data is not None else None, headers=headers)
    try:
        with urlopen(req, timeout=3) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_loopback_api_requires_token_and_rejects_browser_origins_and_other_hosts(service):
    assert request(service, "/api/state", authenticated=False)[0] == 401
    assert request(service, "/api/state", origin="https://unrelated.invalid")[0] == 403
    assert request(service, "/api/state", host="unrelated.invalid")[0] == 403
    code, state = request(service, "/api/state")
    assert code == 200 and state["mode"] == "live"


def test_api_action_loop_and_duplicate_feedback(service):
    code, state = request(service, "/api/mode", {"mode": "rehearsal"})
    assert code == 200
    data = {"proposal_id": state["proposal"]["id"], "outcome": "executed"}
    assert request(service, "/api/feedback", data)[0] == 200
    assert request(service, "/api/feedback", data)[0] == 409
    code, after = request(service, "/api/next-frame", {})
    assert code == 200 and after["proposal"]["title"] == "打出随从"
    assert request(service, "/api/mode", {"mode": "live"})[1]["proposal"] is None


@pytest.mark.parametrize("payload", [[], "PRIVATE_ACCOUNT", {"mode": []}, {"mode": "invalid"}])
def test_bad_requests_do_not_echo_input_or_change_live_state(service, payload):
    code, result = request(service, "/api/mode", payload)
    assert code in (400, 409)
    assert "PRIVATE_ACCOUNT" not in json.dumps(result)
    assert service[1].state()["mode"] == "live"


class LiveFeed:
    def __init__(self):
        self.snap = {"snapshot_id": "first", "status": "snapshot", "game_status": "running", "entities": [],
            "freshness": {"status": "recent"}, "source": {"path": "/fixture/Power.log", "device": 1, "inode": 1, "game_start_line": 1}}
    def sample(self, force=False):
        return {"running": True, "snapshot": deepcopy(self.snap)}


class LiveBridge:
    def sample(self, snap, enabled=True):
        from tools.client_actions import frame_identity
        return {"status": "inferred", "policy": {"update": 15}, "proposal": {
            "frame_id": frame_identity(snap), "action": 3, "kind": "upgrade", "instruction": "升级一次酒馆。",
            "inference_ms": 8., "binding": {"cost": 4}}} if enabled else {"status": "waiting_for_game"}


def test_live_proposal_expiry_never_reissues_same_frame_and_game_end_clears_it():
    clock, feed = [100.], LiveFeed()
    session = WindowSession(feed, clock=lambda: clock[0], ttl=10, model_bridge=LiveBridge())
    old = session.state()["proposal"]
    assert old and session.state()["proposal"]["id"] == old["id"]
    clock[0] = 111.
    assert session.state()["proposal"] is None
    assert session.state()["proposal"] is None
    feed.snap["snapshot_id"] = "second"
    assert session.state()["proposal"] is not None
    feed.snap["game_status"] = "complete"
    assert session.state()["proposal"] is None


def test_race_selection_changes_ui_revision_survives_restart_and_is_scoped_to_game(tmp_path):
    feed = LiveFeed()
    receipts = tmp_path / "sessions" / "first" / "feedback.jsonl"
    session = WindowSession(feed, receipt_file=receipts)
    initial = session.state()
    races = ["UNDEAD", "ELEMENTAL", "QUILBOAR", "MURLOC", "DRAGON"]
    selected = session.set_lobby_races(initial["game_key"], races, "visible_hero_banner")
    assert initial["revision"] != selected["revision"]
    assert not selected["needs_lobby_races"] and selected["lobby_races"] == sorted(races)
    restarted = WindowSession(feed, receipt_file=tmp_path / "sessions" / "second" / "feedback.jsonl")
    assert restarted.state()["lobby_races"] == sorted(races)
    assert restarted.lobby_races_source == "visible_hero_banner"
    feed.snap["source"]["game_start_line"] = 20000
    assert restarted.state()["needs_lobby_races"]
    assert restarted.state()["lobby_races"] is None
    with pytest.raises(WindowError, match="对局已切换"):
        restarted.set_lobby_races(initial["game_key"], races)


def test_live_feed_rechecks_a_settling_menu_without_new_file_bytes(tmp_path, monkeypatch):
    import tools.window_session as module
    path = tmp_path / "Power.log"
    path.write_text("same bytes\n")
    monkeypatch.setattr(module, "newest_log", lambda _: path)
    monkeypatch.setattr(module, "client_running", lambda: True)
    calls = []
    class Reader:
        def read(self, path):
            calls.append(path)
            return {"last_options": {"id": 1}, "options_packet_complete": len(calls) > 1}
    feed = module.LogFeed(tmp_path, None)
    feed.reader = Reader()
    assert not feed.sample()["snapshot"]["options_packet_complete"]
    assert feed.sample()["snapshot"]["options_packet_complete"]
    feed.sample()
    assert len(calls) == 2
