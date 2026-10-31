"""Read client scene and identity through the installed HearthMirror.

Power.log alone supplies card entities, positions, choices and event history.
No accounts, credentials, deck contents or opponents' hands cross this boundary.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time


SCENES = {1: "启动界面", 2: "登录界面", 3: "主界面", 4: "对局界面", 5: "收藏界面",
          6: "开包界面", 7: "对战模式", 8: "好友对战", 10: "竞技场", 13: "冒险模式",
          14: "乱斗模式", 15: "酒馆战棋大厅", 16: "模式选择", 18: "酒馆收藏", 30: "日志界面"}
ERRORS = {"client_not_running", "multiple_clients", "framework_missing", "framework_signature_invalid",
          "framework_load_failed", "framework_api_incompatible", "helper_missing", "memory_access_denied",
          "memory_not_ready", "memory_read_failed", "helper_timeout", "helper_failed"}


class MirrorTransport:
    """Single-owner, bounded JSON pipe to our private helper process."""
    def __init__(self, helper, timeout=8):
        self.helper, self.timeout = Path(helper), timeout
        self.process, self.buffer = None, b""

    def close(self):
        process, self.process = self.process, None
        self.buffer = b""
        if process:
            if process.poll() is None:
                process.kill()  # Only our helper, never Hearthstone.
            process.wait(timeout=2)
            for stream in (process.stdin, process.stdout):
                if stream: stream.close()

    def read(self, game_key, *, scope=None):
        schema = "autotavern.memory.v1" if scope == "public_state" else "autotavern.races.v1"
        if not self.helper.is_file():
            return {"schema": schema, "game_key": game_key, "status": "helper_missing"}
        if not self.process or self.process.poll() is not None:
            self.close()
            self.process = subprocess.Popen([str(self.helper)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, bufsize=0)
        request = {"game_key": game_key}
        if scope: request["scope"] = scope
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError()
            chunk = os.read(self.process.stdout.fileno(), 8192)
            if not chunk or len(self.buffer) + len(chunk) > 16384:
                raise ValueError("Invalid reader response")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)


def validate_public_read(value, game_key, log_session):
    """Accept scene and identity only, including when an older helper sends cards."""
    if not isinstance(value, dict) or value.get("schema") != "autotavern.memory.v1" or value.get("game_key") != game_key:
        return {"status": "invalid_memory_response"}
    status = value.get("status")
    if status != "public_read":
        return {"status": status if isinstance(status, str) and status in ERRORS else "invalid_memory_response"}
    if value.get("log_session") != log_session:
        return {"status": "session_mismatch"}
    if (type(value.get("pid")) is not int or value["pid"] <= 0 or type(value.get("game_handle")) is not int
            or value["game_handle"] < 0 or not isinstance(value.get("public_state"), dict)):
        return {"status": "invalid_memory_response"}
    raw, public = value["public_state"], {}
    scene = raw.get("scene")
    if (isinstance(scene, dict) and type(scene.get("mode")) is int and 0 <= scene["mode"] <= 100
            and type(scene.get("loaded")) is bool and type(scene.get("transitioning")) is bool):
        public["scene"] = {k: scene[k] for k in ("mode", "loaded", "transitioning")}
    if type(raw.get("local_player_id")) is int and 0 < raw["local_player_id"] <= 32:
        public["local_player_id"] = raw["local_player_id"]
    if type(raw.get("spectator")) is bool:
        public["spectator"] = raw["spectator"]
    return {"status": "ready", "pid": value["pid"], "game_handle": value["game_handle"],
            "log_session": log_session, "game_type": value.get("game_type") if type(value.get("game_type")) is int else None,
            "public_state": public}


class PublicStateReader:
    """Continuous background reads; stale/late results never update the model."""
    def __init__(self, helper, *, diagnostic_file=None, timeout=2, interval=.1, max_age=.75, read=None):
        self.transport = MirrorTransport(helper, timeout)
        self.read = read or (lambda key: self.transport.read(key, scope="public_state"))
        self.diagnostic_file = Path(diagnostic_file) if diagnostic_file else None
        self.interval, self.max_age, self.timeout = interval, max_age, timeout
        self.lock = threading.Condition()
        self.desired, self.result, self.completed_at = None, {"status": "idle"}, 0.
        self.closed = False
        self.thread = threading.Thread(target=self._work, name="client-memory", daemon=True)
        self.thread.start()

    def sample(self, game_key, source, *, enabled=True):
        desired = (game_key or "", Path(source["path"]).parent.name) if enabled and source.get("path") else None
        with self.lock:
            if desired != self.desired:
                self.desired = desired
                self.result, self.completed_at = {"status": "reading" if desired else "idle", "game_key": game_key}, 0.
                self.lock.notify_all()
            result = deepcopy(self.result)
            if result.get("status") == "ready" and time.monotonic() - self.completed_at > self.max_age:
                return {"status": "stale", "game_key": game_key}
            return result

    def _work(self):
        active, previous, previous_identity, last_recorded, next_read = None, None, None, None, 0.
        while True:
            with self.lock:
                if self.closed: break
                if self.desired != active:
                    active, previous, previous_identity, last_recorded, next_read = self.desired, None, None, None, 0.
                if not active or time.monotonic() < next_read:
                    self.lock.wait(timeout=self.interval)
                    continue
            started = time.monotonic()
            try:
                result = validate_public_read(self.read(active[0]), *active)
            except TimeoutError:
                result = {"status": "helper_timeout"}
                self.transport.close()
            except (OSError, ValueError):
                result = {"status": "helper_failed"}
                self.transport.close()
            completed = time.monotonic()
            signature = json.dumps(result, sort_keys=True)
            public = result.get("public_state", {})
            identity = (result.get("status"), result.get("pid"), result.get("log_session"), result.get("game_handle"),
                        public.get("local_player_id"), public.get("spectator"))
            result.update(game_key=active[0], source="hearthmirror", stable=signature == previous,
                          identity_stable=identity == previous_identity,
                          elapsed_ms=round((completed-started)*1000, 3))
            with self.lock:
                if self.closed or self.desired != active: continue
                previous, previous_identity = signature, identity
                self.result, self.completed_at = result, completed
                next_read = completed + (self.interval if result["status"] == "ready" else .5)
                record_signature = (signature, result["stable"], result["identity_stable"])
                if self.diagnostic_file and record_signature != last_recorded:
                    try:
                        self.diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
                        fd = os.open(self.diagnostic_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                        with os.fdopen(fd, "w") as stream:
                            os.fchmod(stream.fileno(), 0o600)
                            stream.write(json.dumps({**result, "at": datetime.now(timezone.utc).isoformat()}) + "\n")
                    except OSError: pass
                    last_recorded = record_signature
        self.transport.close()

    def close(self):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        self.thread.join(timeout=self.timeout + 3)


def merge_public_state(snapshot, memory, evidence=None):
    """Apply client identity and scene; preserve all log entities and choices."""
    diagnostics = {"status": memory.get("status", "idle"), "source": "hearthmirror"}
    if memory.get("status") != "ready": return snapshot, None, diagnostics
    from .client_common import log_game_key
    source = snapshot.get("source", {})
    key = log_game_key(source) or ""
    if (memory.get("game_key") != key or not source.get("path")
            or memory.get("log_session") != Path(source["path"]).parent.name):
        diagnostics["status"] = "context_mismatch"
        return snapshot, None, diagnostics
    public = memory.get("public_state", {})
    scene = public.get("scene", {})
    loaded = scene.get("loaded") is True and scene.get("transitioning") is False
    scene_label = SCENES.get(scene.get("mode")) if loaded else None
    expected = evidence or {}
    matched = bool(expected) and all(memory.get(k) == expected.get(k) for k in ("pid", "game_handle", "log_session"))
    diagnostics.update(elapsed_ms=memory.get("elapsed_ms"), game_bound=matched, stable=memory.get("stable", False))
    diagnostics["blocks_actions"] = bool(scene and (not loaded or scene.get("mode") != 4)) or public.get("spectator") is True or bool(expected and not matched)
    if not matched or memory.get("game_type") not in (23, 24) or not loaded or scene.get("mode") != 4:
        return snapshot, scene_label, diagnostics
    if public.get("spectator") is not False:
        diagnostics["perspective"] = "spectator_or_unknown"
        return snapshot, scene_label, diagnostics
    pid = public.get("local_player_id")
    entities = snapshot.get("entities", [])
    players = [e for e in entities if e.get("type") == "PLAYER" and e.get("tags", {}).get("PLAYER_ID") == pid]
    if type(pid) is not int or len(players) != 1:
        diagnostics["perspective"] = "player_not_in_log"
        return snapshot, scene_label, diagnostics
    # A single read can describe a transition; accept identity after two equal reads.
    if not memory.get("identity_stable"):
        diagnostics["perspective"] = "confirming"
        return snapshot, scene_label, diagnostics
    merged = {**snapshot, "perspective": {"player_id": pid, "source": "hearthmirror"}}
    if "validation" in snapshot:
        merged["validation"] = {**snapshot["validation"], "issues": [i for i in snapshot["validation"].get("issues", [])
                                                                      if i.get("code") != "perspective_unconfirmed"]}
    if "model_blockers" in snapshot:
        merged["model_blockers"] = [b for b in snapshot["model_blockers"] if b != "perspective_unconfirmed"]
    diagnostics["perspective"] = "hearthmirror"
    return merged, scene_label, diagnostics
