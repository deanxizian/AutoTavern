"""Recording must preserve log generations and never attach previous-game metadata."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.client_common import log_game_key
from tools.client_record import LogArchive, snapshot_with_window_context


@pytest.mark.parametrize("reset", ["truncate", "rewrite", "replace"])
def test_log_rotation_keeps_distinct_copies_and_resumes_each_offset(tmp_path, reset):
    source = tmp_path / "session" / "Power.log"
    source.parent.mkdir()
    source.write_bytes(b"first\n")
    archive = LogArchive(tmp_path / "recording")
    assert archive.capture(source)
    with source.open("ab") as stream:
        stream.write(b"append\n")
    assert archive.capture(source)
    original = source.read_bytes()
    if reset == "replace":
        replacement = source.with_suffix(".new")
        replacement.write_bytes(b"second\n")
        replacement.replace(source)
    elif reset == "rewrite":
        # A truncation that regrew before the next poll needs the byte anchor.
        source.write_bytes(b"x" * len(original))
    else:
        source.write_bytes(b"new\n")
    assert archive.capture(source)
    for chunk in (b"more\n", b"and more\n"):
        with source.open("ab") as stream:
            stream.write(chunk)
        assert archive.capture(source)
    assert archive.capture(source)  # Repeated idle polls cannot duplicate bytes.
    manifest = archive.manifest()
    assert len(manifest) == 2
    assert [Path(row["copy"]).read_bytes() for row in manifest] == [original, source.read_bytes()]
    for row in manifest:
        path = Path(row["copy"])
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.stat().st_mode & 0o777 == 0o600


def test_missing_empty_and_repeated_truncations_have_unique_archive_generations(tmp_path):
    source = tmp_path / "Power.log"
    archive = LogArchive(tmp_path / "recording")
    assert not archive.capture(source)
    for _ in range(3):
        source.write_bytes(b"")
        assert archive.capture(source)
        assert archive.capture(source)
        source.write_bytes(b"game\n")
        assert archive.capture(source)
    rows = archive.manifest()
    assert len({row["copy"] for row in rows}) == 3
    assert all(Path(row["copy"]).read_bytes() == b"game\n" for row in rows)


def context():
    snapshot = {
        "source": {"path": "/logs/Power.log", "device": 1, "inode": 2, "game_start_line": 3},
        "entities": [{"entity_id": 8, "type": "PLAYER", "tags": {"PLAYER_ID": 3}}],
        "last_choices": {"entity_ids": [101, 201]},
        "perspective": {"player_id": 11, "source": "log"},
        "validation": {"issues": [{"code": "perspective_unconfirmed"}, {"code": "other"}]},
        "model_blockers": ["perspective_unconfirmed", "other"],
    }
    window = {
        "game_key": log_game_key(snapshot["source"]),
        "lobby_races": ["BEAST", "DEMON", "DRAGON", "NAGA", "QUILBOAR"],
        "lobby_races_source": "hearthmirror",
        "perspective": {"player_id": 3, "source": "hearthmirror"},
    }
    return snapshot, window


def test_recorder_attaches_current_game_context_without_mutating_reader_cache():
    snapshot, window = context()
    original = deepcopy(snapshot)
    result = snapshot_with_window_context(snapshot, window)
    assert snapshot == original
    assert result["lobby_races"] == window["lobby_races"]
    assert result["lobby_races"] is not window["lobby_races"]
    assert result["lobby_races_source"] == "hearthmirror"
    assert result["perspective"] == window["perspective"]
    assert result["model_blockers"] == ["other"]
    assert result["validation"]["issues"] == [{"code": "other"}]
    assert result["entities"] is snapshot["entities"]
    assert result["last_choices"] is snapshot["last_choices"]


@pytest.mark.parametrize("field,value", [("game_start_line", 4), ("inode", 9), ("path", "/logs/next/Power.log")])
def test_recorder_rejects_metadata_from_a_different_game_or_log(field, value):
    snapshot, window = context()
    snapshot["source"][field] = value
    assert snapshot_with_window_context(snapshot, window) is snapshot


@pytest.mark.parametrize("perspective", [
    {"player_id": 4, "source": "hearthmirror"}, {"player_id": 3, "source": "log"}, {},
])
def test_recorder_does_not_adopt_unknown_or_inferred_window_identity(perspective):
    snapshot, window = context()
    window["perspective"] = perspective
    result = snapshot_with_window_context(snapshot, window)
    assert result["perspective"] == snapshot["perspective"]
    assert result["validation"] == snapshot["validation"]
    assert result["model_blockers"] == snapshot["model_blockers"]


def test_completed_recording_finalizes_when_client_quits_during_final_combat(tmp_path, monkeypatch):
    pytest.importorskip("hslog")
    from tools import client_record, client_features, client_stream

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "Power.log").write_bytes(b"recorded game\n")
    connections = tmp_path / "connections" / "session"
    connections.mkdir(parents=True)
    (connections / "connection.json").write_text(json.dumps({"url": "http://localhost", "token": "test-token"}))
    now = [0.]
    monkeypatch.setattr(client_record, "time", SimpleNamespace(
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    monkeypatch.setattr(client_record, "urlopen", lambda *args, **kwargs: io.StringIO(json.dumps({
        "revision": now[0], "client_running": now[0] == 0,
    })))
    monkeypatch.setattr(client_stream, "IncrementalReader", lambda: SimpleNamespace(read=lambda path: {
        "entities": [{"entity_id": 1}], "client_phase": "combat",
        "game_status": "running" if now[0] == 0 else "complete",
    }))
    packet = SimpleNamespace(gaps=[], actions=SimpleNamespace(player={}, phase="game_over"), report=lambda: {})
    monkeypatch.setattr(client_features, "ClientFeatureAdapter", lambda: SimpleNamespace(convert=lambda snapshot: packet))
    output = tmp_path / "recording"
    client_record.record(logs, output, connections.parent, duration=30)
    assert now[0] < 4  # Must not wait for the full timeout after the client exits.
    manifest = json.loads((output / "recording-end.json").read_text())
    assert manifest["game_complete"] and manifest["frames"] == 2
    assert len(manifest["files"]) == 1
    assert Path(manifest["files"][0]["copy"]).read_bytes() == b"recorded game\n"
