"""Nonblocking, per-game lobby race reader using the installed macOS HearthMirror.

The native helper is isolated from both the UI and model process. It exposes no
screen capture, game input, accounts, connection credentials or hidden boards.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

from .client_memory import MirrorTransport

RACES = {11: "UNDEAD", 14: "MURLOC", 15: "DEMON", 17: "MECHANICAL", 18: "ELEMENTAL",
         20: "BEAST", 23: "PIRATE", 24: "DRAGON", 43: "QUILBOAR", 92: "NAGA", 126: "ABERRATION"}
SOURCE = "hearthmirror"
MESSAGES = {
    "reading": "正在自动读取本局种族。",
    "confirming": "已读到种族，正在核对本局信息。",
    "ready": "已自动读取本局种族。",
    "idle": "新对局开始后自动读取种族。",
    "client_not_running": "打开炉石后将自动读取本局种族。",
    "multiple_clients": "检测到多个炉石进程，暂时无法确定当前对局。",
    "framework_missing": "未找到 HSTracker 的读取组件，请将 HSTracker 安装到应用程序目录。",
    "framework_signature_invalid": "HSTracker 读取组件的签名校验失败，请重新安装官方版本。",
    "framework_load_failed": "无法加载 HSTracker 读取组件，请更新 HSTracker。",
    "framework_api_incompatible": "当前 HSTracker 组件接口不兼容，请更新 HSTracker。",
    "helper_missing": "种族读取组件未构建，请重新启动助手。",
    "memory_access_denied": "macOS 未允许读取炉石进程，种族读取已暂停；详细原因已记录。",
    "memory_not_ready": "等待客户端准备本局种族，助手会自动重试。",
    "memory_read_failed": "本次客户端读取未成功，正在自动重试。",
    "helper_timeout": "客户端读取超时，正在重新连接读取组件。",
    "helper_failed": "读取组件意外退出，正在自动重试。",
    "invalid_memory_response": "客户端种族数据尚不完整，正在自动重试。",
    "session_mismatch": "等待读取组件切换到当前日志会话。",
    "stale_memory_game": "等待读取组件切换到新对局。",
    "not_battlegrounds": "等待客户端进入酒馆战棋对局。",
}


def validate_read(value, game_key, log_session):
    """All IDs must be known; never silently drop a new/unknown race."""
    if not isinstance(value, dict) or value.get("schema") != "autotavern.races.v1" or value.get("game_key") != game_key:
        return "invalid_memory_response", None
    status = value.get("status")
    if status != "read":
        allowed_error = (isinstance(status, str) and status in MESSAGES
                         and status not in ("reading", "confirming", "ready", "idle"))
        return (status if allowed_error else "invalid_memory_response"), None
    if value.get("log_session") != log_session:
        return "session_mismatch", None
    # The model adapter supports solo lobbies, not a fabricated solo view of Duos.
    if type(value.get("game_type")) is not int or value["game_type"] not in (23, 24):
        return "not_battlegrounds", None
    ids = value.get("race_ids")
    if (not isinstance(ids, list) or len(ids) != 5 or any(type(i) is not int or i not in RACES for i in ids)
            or len(set(ids)) != 5 or type(value.get("game_handle")) is not int or value["game_handle"] <= 0
            or type(value.get("pid")) is not int or value["pid"] <= 0):
        return "invalid_memory_response", None
    return "ready", sorted(RACES[i] for i in ids)


class RaceReader:
    def __init__(self, helper, *, diagnostic_file=None, timeout=8, read=None, retry_scale=1):
        self.helper = Path(helper)
        self.diagnostic_file = Path(diagnostic_file) if diagnostic_file else None
        self.timeout, self.retry_scale = timeout, retry_scale
        self.read = read or self._read
        self.lock = threading.Condition()
        self.desired = None
        self.result = {"status": "idle", "detail": MESSAGES["idle"]}
        self.closed = False
        self.transport = MirrorTransport(helper, timeout)
        self.bindings = {}
        self.thread = threading.Thread(target=self._work, name="lobby-races", daemon=True)
        self.thread.start()

    def sample(self, game_key, source, *, enabled):
        desired = (game_key, str(Path(source["path"]).parent.name)) if enabled and game_key and source.get("path") else None
        with self.lock:
            if desired != self.desired:
                self.desired = desired
                status = "reading" if desired else "idle"
                self.result = {"game_key": game_key, "status": status, "detail": MESSAGES[status]}
                self.lock.notify_all()
            return deepcopy(self.result)

    def restore_bindings(self, records):
        with self.lock:
            for key, record in records.items():
                evidence = record.get("evidence", {}) if isinstance(record, dict) else {}
                pid, session, handle = (evidence.get(k) for k in ("pid", "log_session", "game_handle"))
                if type(pid) is int and type(handle) is int and isinstance(session, str):
                    self.bindings[(pid, session, handle)] = key
            self.bindings = dict(list(self.bindings.items())[-16:])

    @property
    def process(self):
        return self.transport.process

    def _stop_helper(self):
        self.transport.close()

    def _read(self, game_key):
        return self.transport.read(game_key)

    def _record(self, result):
        if not self.diagnostic_file:
            return
        self.diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
        # Explicit allowlist: never persist arbitrary framework output.
        event = {k: v for k, v in result.items() if k in (
            "game_key", "status", "attempt", "elapsed_ms", "pid", "game_handle", "log_session", "races",
            "race_ids", "game_type", "memory_access", "mach_error")}
        event["at"] = datetime.now(timezone.utc).isoformat()
        fd = os.open(self.diagnostic_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _work(self):
        active, candidate, attempt, next_read, last_recorded = None, None, 0, 0., None
        while True:
            with self.lock:
                if self.closed:
                    break
                desired = self.desired
                if desired != active:
                    active, candidate, attempt, next_read, last_recorded = desired, None, 0, 0., None
                ready = self.result.get("status") == "ready"
                if not active or ready or time.monotonic() < next_read:
                    self.lock.wait(timeout=.1)
                    continue
            started = time.monotonic()
            attempt += 1
            value = {}
            try:
                value = self.read(active[0])
                status, races = validate_read(value, *active)
                if not isinstance(value, dict):
                    value = {}
            except TimeoutError:
                status, races = "helper_timeout", None
                self._stop_helper()
            except (OSError, ValueError):
                status, races = "helper_failed", None
                self._stop_helper()
            result = {"game_key": active[0], "status": status, "source": SOURCE, "attempt": attempt,
                      "elapsed_ms": round((time.monotonic()-started)*1000, 2)}
            for name in ("pid", "game_handle", "game_type", "mach_error"):
                if type(value.get(name)) is int:
                    result[name] = value[name]
            if type(value.get("memory_access")) is bool:
                result["memory_access"] = value["memory_access"]
            if isinstance(value.get("log_session"), str) and len(value["log_session"]) < 128:
                result["log_session"] = value["log_session"]
            if (isinstance(value.get("race_ids"), list) and len(value["race_ids"]) <= 16
                    and all(type(i) is int for i in value["race_ids"])):
                result["race_ids"] = value["race_ids"]
            # Bind a fresh handle to exactly one log game, and require two stable
            # reads. Late results from a previous game/disabled mode are discarded.
            with self.lock:
                if self.closed or self.desired != active:
                    continue
                if status == "ready":
                    identity = (value["pid"], value["log_session"], value["game_handle"])
                    confirmation = (identity, tuple(races))
                    if self.bindings.get(identity, active[0]) != active[0]:
                        result["status"], candidate = "stale_memory_game", None
                    elif candidate != confirmation:
                        candidate = confirmation
                        result["status"] = "confirming"
                    else:
                        self.bindings[identity] = active[0]
                        self.bindings = dict(list(self.bindings.items())[-16:])
                        result["races"] = races
                    for name in ("pid", "game_handle", "log_session"):
                        result[name] = value[name]
                else:
                    candidate = None
                result["detail"] = MESSAGES[result["status"]]
                self.result = result
                delay = .25 if result["status"] == "confirming" else min(2., .25 * 2**min(attempt-1, 4))
                next_read = time.monotonic() + delay * self.retry_scale
                signature = (result["status"], tuple(result.get("races", [])), result.get("mach_error"))
                if signature != last_recorded or attempt % 10 == 0:
                    try: self._record(result)
                    except OSError: pass
                    last_recorded = signature
        self._stop_helper()

    def close(self):
        with self.lock:
            self.closed = True
            self.lock.notify_all()
        # Wait for an in-flight read's bounded deadline before cleaning up its pipes.
        self.thread.join(timeout=self.timeout + 3)
