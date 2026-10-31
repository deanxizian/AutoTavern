"""Record one live game and the companion's actual output, without game input."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

from .client_common import log_game_key, newest_log, write_private_json


class LogArchive:
    """Keep separate byte-exact copies when a client log is replaced or truncated."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.active = {}
        self.files = []

    def capture(self, source):
        source = Path(source)
        previous = self.active.get(source)
        try:
            with source.open("rb") as stream:
                stat = os.fstat(stream.fileno())
                identity = (stat.st_dev, stat.st_ino)
                offset = 0
                if previous and previous["identity"] == identity and stat.st_size >= previous["offset"]:
                    offset = previous["offset"]
                    stream.seek(offset - len(previous["anchor"]))
                    if stream.read(len(previous["anchor"])) != previous["anchor"]:
                        offset = 0
                        previous = None
                else:
                    previous = None
                stream.seek(offset)
                data = stream.read(stat.st_size - offset)
                if len(data) != stat.st_size - offset or os.fstat(stream.fileno()).st_size < stat.st_size:
                    # Retry a concurrent truncation without adding a partial generation.
                    return False
        except (FileNotFoundError, NotADirectoryError):
            return False

        if previous is None:
            target = self.directory / source.parent.name / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            generation = 1
            while target.exists():
                generation += 1
                target = target.with_name(f"{source.stem}-{generation:03}{source.suffix}")
            with target.open("xb") as stream:
                os.fchmod(stream.fileno(), 0o600)
            self.files.append({"source": str(source), "copy": str(target)})
            previous = {"identity": identity, "offset": 0, "anchor": b"", "target": target}
            self.active[source] = previous
        if data:
            with previous["target"].open("ab") as stream:
                stream.write(data)
            previous["offset"] = offset + len(data)
            previous["anchor"] = (previous["anchor"] + data)[-64:]
        return True

    def manifest(self):
        rows = []
        for row in self.files:
            path = Path(row["copy"])
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append({**row, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
        return rows


def snapshot_with_window_context(snapshot, window):
    """Only attach metadata that the companion confirmed for this exact game."""
    key = log_game_key(snapshot.get("source", {}))
    if not key or key != window.get("game_key"):
        return snapshot
    result = dict(snapshot)
    if window.get("lobby_races"):
        result["lobby_races"] = list(window["lobby_races"])
        result["lobby_races_source"] = window.get("lobby_races_source") or "companion_per_game_confirmation"
    perspective = window.get("perspective") or {}
    player_id = perspective.get("player_id")
    if perspective.get("source") == "hearthmirror" and any(
        entity.get("type") == "PLAYER" and entity.get("tags", {}).get("PLAYER_ID") == player_id
        for entity in snapshot.get("entities", [])
    ):
        result["perspective"] = {"player_id": player_id, "source": "hearthmirror"}
        result["validation"] = {**snapshot.get("validation", {}), "issues": [
            issue for issue in snapshot.get("validation", {}).get("issues", [])
            if issue.get("code") != "perspective_unconfirmed"
        ]}
        result["model_blockers"] = [
            code for code in snapshot.get("model_blockers", []) if code != "perspective_unconfirmed"
        ]
    return result


def record(logs, output, connections, duration=5400):
    from .client_actions import frame_identity
    from .client_features import ClientFeatureAdapter
    from .client_stream import IncrementalReader

    output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    reader, adapter = IncrementalReader(), ClientFeatureAdapter()
    archive = LogArchive(output / "raw")
    sequence, last_frame, last_window = 0, None, None
    packet, report = None, {}
    started, saw_game, complete_at = time.monotonic(), False, None
    windows, gaps = Counter(), Counter()
    def append(name, value):
        path = output/name
        with path.open("a", encoding="utf-8") as f:
            os.fchmod(f.fileno(), 0o600)
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
    write_private_json(output/"recording-start.json", {"started_at": datetime.now(timezone.utc).isoformat(), "logs": str(logs), "scope": "live_read_only_log_and_window_recording"})
    print(json.dumps({"recording": str(output), "status": "started"}), flush=True)
    try:
        while time.monotonic()-started < duration:
            now = datetime.now(timezone.utc).isoformat()
            path = newest_log(logs)
            if path:
                for source in (path, path.parent/"LoadingScreen.log"):
                    archive.capture(source)
            window = {}
            try:
                candidates = sorted(connections.glob("*/connection.json"), key=lambda p:p.stat().st_mtime_ns, reverse=True)
                if candidates:
                    c = json.loads(candidates[0].read_text())
                    with urlopen(Request(c["url"]+"/api/state", headers={"Authorization":"Bearer "+c["token"]}), timeout=2) as response:
                        window = json.load(response)
                    stamp = (str(candidates[0]), window.get("revision"))
                    if stamp != last_window:
                        append("window.jsonl", {"at": now, "state": window})
                        windows[window.get("model_status", {}).get("title", "unknown")] += 1
                        last_window = stamp
            except (OSError, ValueError, KeyError, TypeError) as error:
                window = {"recording_error": type(error).__name__}
            snapshot = snapshot_with_window_context(reader.read(path), window)
            frame = frame_identity(snapshot)
            if frame != last_frame:
                sequence += 1
                packet = adapter.convert(snapshot)
                report = packet.report()
                gaps.update(g["code"] for g in packet.gaps)
                with gzip.open(output/f"snapshot-{sequence:05}.json.gz", "wt", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, allow_nan=False)
                os.chmod(output/f"snapshot-{sequence:05}.json.gz", 0o600)
                append("timeline.jsonl", {"sequence": sequence, "at": now, "source_bytes": snapshot.get("source", {}).get("bytes"),
                    "turn": packet.actions.player.get("tags", {}).get("TURN"), "phase": packet.actions.phase,
                    "status": snapshot.get("status"), "performance": snapshot.get("performance"), "input": report,
                    "action_counts": snapshot.get("action_counts"), "window_ready": window.get("live_ready"),
                    "window_model": window.get("model_status"), "instruction": (window.get("proposal") or {}).get("instruction")})
                last_frame = frame
            saw_game |= bool(snapshot.get("entities")) and snapshot.get("game_status") == "running"
            latest = {"at": now, "sequence": sequence, "game_status": snapshot.get("game_status"),
                "client_phase": snapshot.get("client_phase"),
                "phase": packet.actions.phase if packet else None,
                "turn": packet.actions.player.get("tags", {}).get("TURN") if packet else None,
                "actions": snapshot.get("action_counts", {}), "model_status": window.get("model_status"),
                "live_ready": window.get("live_ready"), "proposal": window.get("proposal"),
                "gaps": report.get("gaps", []), "source": snapshot.get("source", {}),
                "game_key": window.get("game_key"), "lobby_races": window.get("lobby_races"),
                "lobby_races_source": window.get("lobby_races_source"), "race_reader": window.get("race_reader"),
                "memory_reader": window.get("memory_reader"), "perspective": window.get("perspective"),
                "client_scene_source": window.get("client_scene_source")}
            write_private_json(output/"latest.json", latest)
            if saw_game and snapshot.get("game_status") == "complete":
                complete_at = complete_at or time.monotonic()
                if time.monotonic()-complete_at > 3 and (
                    snapshot.get("client_phase") != "combat" or window.get("client_running") is False
                ):
                    break
            time.sleep(.15)
    except KeyboardInterrupt:
        pass
    finally:
        write_private_json(output/"recording-end.json", {"ended_at": datetime.now(timezone.utc).isoformat(),
            "files": archive.manifest(), "frames": sequence, "game_complete": bool(complete_at),
            "window_status_changes": dict(windows), "observed_gap_counts": dict(gaps)})
        print(json.dumps({"recording":str(output), "frames":sequence, "game_complete":bool(complete_at)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--connections", type=Path, default=Path("runs/client-window/sessions"))
    args = p.parse_args()
    record(args.logs, args.out.resolve(), args.connections.resolve())
