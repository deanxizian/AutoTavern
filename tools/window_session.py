"""State for the native companion window. Inference runs in a separate process."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

if __package__:
    from .client_audit import app_info, client_running
    from .client_common import load_catalog, newest_log
    from .client_projection import project_live_cards
    from .client_model_bridge import model_summary
    from .manual_advice import ACTION_LAYOUT, render_or_block
else:
    from client_audit import app_info, client_running
    from client_common import load_catalog, newest_log
    from client_projection import project_live_cards
    from tools.client_model_bridge import model_summary
    from manual_advice import ACTION_LAYOUT, render_or_block


TITLES = {"buy": "购买卡牌", "play_minion": "打出随从", "freeze": "冻结酒馆",
          "unfreeze": "解冻酒馆", "wait_for_combat": "等待战斗", "sell": "出售随从",
          "move": "调整站位", "target": "选择目标", "magnetize": "磁力合体",
          "upgrade": "升级酒馆", "refresh": "刷新酒馆", "cast": "施放法术",
          "activate": "发动随从技能", "hero_power": "使用英雄技能", "dark_gift": "使用黑暗之赐",
          "choose_hero": "选择英雄", "choose_hero_power": "选择英雄技能", "choose_trinket": "选择饰品",
          "choose_discover": "选择发现卡牌"}
ZONE_LABELS = {"board": "战场", "hand": "手牌", "shop": "酒馆"}
SCENE_LABELS = {"STARTUP": "启动界面", "LOGIN": "登录界面", "HUB": "主界面",
                "JOURNAL": "日志界面", "PACKOPENING": "开包界面", "BACON": "酒馆战棋大厅", "GAMEPLAY": "对局界面"}
OPERATION_LABELS = {"buy_minion": "购买随从", "buy_spell": "购买酒馆法术", "play_minion": "打出随从",
    "sell": "出售随从", "upgrade": "升级酒馆", "refresh": "刷新酒馆", "freeze_toggle": "冻结／解冻",
    "board_position": "调整战场站位", "shop_position": "调整酒馆位置", "activate_minion": "发动随从",
    "dark_gift": "暗黑赐礼", "cast_spell": "施放法术", "choice": "提交选择", "hero_power": "使用英雄技能"}


def latest_scene(path):
    """Last completed LoadingScreen mode, not a game state or a current-screen proof."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 1024 * 1024))
            if size > 1024 * 1024:
                stream.readline()
            raw = stream.read()
            data = raw[:raw.rfind(b"\n") + 1]
        modes = re.findall(rb"\bcurrMode\s*[=:]\s*([A-Z_]+)\b", data)
        return SCENE_LABELS.get(modes[-1].decode("ascii")) if modes else None
    except OSError:
        return None


class WindowError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class LogFeed:
    """Read only when bytes/identity/age status changes, and cache process checks."""
    def __init__(self, logs, client_app):
        self.logs, self.client_app = logs, client_app
        self.signature = object()
        self.cached = None
        self.running = None
        self.process_checked = -10.0
        self.build = app_info(client_app)["build"] if client_app else None
        if __package__:
            from .client_stream import IncrementalReader
        else:
            from client_stream import IncrementalReader
        self.reader = IncrementalReader(installed_build=self.build)
        self.scene_signature = None
        self.scene = None

    def sample(self, force=False):
        if __package__:
            from .client_state import watch_signature
        else:
            from client_state import watch_signature
        try:
            path = newest_log(self.logs) if self.logs else None
            signature = watch_signature(path)
            cached = self.cached or {}
            choices = cached.get("last_choices") or {}
            settling = (cached.get("last_options") and cached.get("options_packet_complete") is False) or (
                choices and not choices.get("selection_sent_or_resolved") and cached.get("choice_packet_complete") is False)
            if force or signature != self.signature or settling:
                self.cached = self.reader.read(path)
                self.signature = object() if self.cached.get("source", {}).get("changed_while_reading") else signature
        except OSError:
            path = None
            self.cached = {"status": "read_or_parse_error", "model_ready": False, "live_ready": False}
            self.signature = object()
        if force or time.monotonic() - self.process_checked > 3:
            self.running = client_running()
            self.process_checked = time.monotonic()
        scene_path = path.parent / "LoadingScreen.log" if path else None
        try:
            scene_stat = scene_path.stat() if scene_path else None
            scene_signature = (scene_path, scene_stat.st_ino, scene_stat.st_size, scene_stat.st_mtime_ns) if scene_stat else None
        except OSError:
            scene_signature = None
        if scene_signature != self.scene_signature:
            self.scene = latest_scene(scene_path) if scene_path else None
            self.scene_signature = scene_signature
        return {"snapshot": self.cached or {}, "running": self.running, "build": self.build,
                "scene": self.scene}


def rehearsal_frame(index):
    """Explicit synthetic frames, never a live-game simulation or policy output."""
    _, catalog = load_catalog()
    def minion(card_id, entity):
        c = catalog[card_id]
        return {"id": card_id, "name_zh": c["name_zh"], "type": "minion", "atk": c["atk"],
                "hp": c["hp"], "golden": False, "eid": entity, "hid": entity, "sid": entity,
                "price": 3, "costs_health": False, "frozen": False}
    first, second = minion("AT_121", 101), minion("BGS_004", 102)
    action = (8, 46, 2, 0)[index]
    mask = [0] * 245
    mask[action] = 1
    me = {"alive": True, "gold": 5 if index == 0 else 2, "tier": 2,
          "board": [first] if index >= 2 else [], "hand": [first] if index == 1 else [],
          "shop": [first, second] if index == 0 else [second], "pending": None}
    if index == 3:
        me["shop"][0]["frozen"] = True
    return {"phase": "recruit", "turn": 4, "actions": dict(ACTION_LAYOUT), "mask": mask,
            "me": me}, action


class WindowSession:
    def __init__(self, feed, *, receipt_file=None, clock=time.time, ttl=60, model_bridge=None, race_reader=None, memory_reader=None):
        self.feed, self.receipt_file, self.clock, self.ttl = feed, receipt_file, clock, ttl
        self.lock = threading.RLock()
        self.mode = "live"
        self.proposal = None
        self.frame = None
        self.index = 0
        self.advance = False
        self.message = None
        self.history = []
        self.model_bridge = model_bridge
        self.race_reader = race_reader
        self.memory_reader = memory_reader
        self.game_key = None
        self.lobby_races = None
        self.lobby_races_source = None
        self.dismissed_frame = None
        self.races_file = Path(receipt_file).parent.parent / "lobby-races.json" if receipt_file else None
        self.race_records = {}
        if self.races_file:
            try: self.race_records = json.loads(self.races_file.read_text())
            except (OSError, ValueError): pass
        if self.race_reader:
            self.race_reader.restore_bindings(self.race_records)
        _, self.catalog = load_catalog()

    def set_lobby_races(self, game_key, races, source="user_confirmed_for_game"):
        allowed = {"BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECHANICAL", "MURLOC", "PIRATE", "QUILBOAR", "UNDEAD", "ABERRATION", "NAGA"}
        with self.lock:
            self.state()
            values = races.split(",") if isinstance(races, str) else races
            if game_key != self.game_key or not self.game_key:
                raise WindowError("stale_game", "对局已切换，请确认新一局的种族。")
            if not isinstance(values, list) or len(values) != 5 or len(set(values)) != 5 or not set(values) <= allowed:
                raise WindowError("invalid_races", "请选择本局显示的 5 个种族。")
            # Memory provenance can only come from the local reader, not this API.
            self._remember_races(values, "user_confirmed_for_game")
            self.dismissed_frame = None
            return self.state()

    def _remember_races(self, races, source, evidence=None):
        self.lobby_races, self.lobby_races_source = sorted(races), source
        self.race_records[self.game_key] = {"races": self.lobby_races, "source": source}
        if evidence:
            self.race_records[self.game_key]["evidence"] = {
                k: evidence[k] for k in ("pid", "game_handle", "log_session") if k in evidence}
        self.race_records = dict(list(self.race_records.items())[-8:])
        if self.races_file:
            from tools.client_common import write_private_json
            write_private_json(self.races_file, self.race_records)

    def _expire(self):
        if self.proposal and self.clock() >= self.proposal["expires_at"]:
            if self.mode == "live": self.dismissed_frame = self.proposal["frame_id"]
            self.proposal = None
            self.advance = False
            self.message = "这条指引已过期，请重新载入局面。"

    def _install(self, view, action):
        rendered = render_or_block(view, action)
        if rendered["status"] != "rendered":
            raise WindowError(rendered["reason"], rendered["message"])
        # Keep only the acting player's projection. Never retain/export private opponents.
        self.frame = {key: deepcopy(view.get(key)) for key in ("phase", "turn", "me")}
        frame_id = uuid.uuid4().hex
        self.proposal = {"id": uuid.uuid4().hex, "frame_id": frame_id,
                         "view_id": rendered["view_id"], "scope": "simulator_action_preview",
                         "title": TITLES.get(rendered["kind"], "下一步操作"),
                         "instruction": rendered["instruction"],
                         "references": rendered["references"], "cost": rendered.get("cost"),
                         "expires_at": self.clock() + self.ttl, "live_ready": False}
        self.message = None
        self.advance = False

    def set_mode(self, mode):
        with self.lock:
            if mode not in ("live", "rehearsal"):
                raise WindowError("invalid_mode", "请选择真实日志或操作演练。")
            self.mode, self.index = mode, 0
            self.proposal = self.frame = self.message = None
            self.advance = False
            if mode == "rehearsal":
                self._install(*rehearsal_frame(0))
            return self.state()

    def publish_preview(self, payload):
        with self.lock:
            if payload.get("scope") != "simulator_action_preview":
                raise WindowError("live_advice_unavailable", "实战建议尚未启用；这里只接受模拟器动作预览。")
            # Validate before changing mode or replacing the current proposal.
            rendered = render_or_block(payload.get("view"), payload.get("action"))
            if rendered["status"] != "rendered":
                raise WindowError(rendered["reason"], rendered["message"])
            self._install(payload["view"], payload["action"])
            self.mode = "replay"
            return self.state()

    def feedback(self, proposal_id, outcome):
        with self.lock:
            self._expire()
            if not self.proposal or proposal_id != self.proposal["id"]:
                raise WindowError("stale_proposal", "指引已变化或过期，请先查看最新局面。")
            if outcome not in ("executed", "not_executed"):
                raise WindowError("invalid_feedback", "这条指引不能记录此反馈。")
            event = {"id": uuid.uuid4().hex, "at": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
                     "mode": self.mode, "proposal_id": proposal_id,
                     "frame_id": self.proposal["frame_id"], "outcome": outcome,
                     "title": self.proposal["title"], "execution_verified": False}
            if self.receipt_file:
                self.receipt_file.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self.receipt_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.history.insert(0, event)
            self.history = self.history[:12]
            if self.mode == "live": self.dismissed_frame = self.proposal["frame_id"]
            self.proposal = None
            self.advance = outcome == "executed"
            self.message = "已记录你的反馈，等待操作后的新局面。" if self.advance else "已标记未执行，当前指引已收起。"
            return self.state()

    def next_frame(self):
        with self.lock:
            self._expire()
            if self.mode != "rehearsal" or self.proposal:
                raise WindowError("awaiting_feedback", "请先反馈当前指引；真实日志不能由演练推进。")
            if self.advance:
                self.index = (self.index + 1) % 4
            self._install(*rehearsal_frame(self.index))
            return self.state()

    def state(self, force=False):
        with self.lock:
            self._expire()
            reading = self.feed.sample(force=force)
            snap = reading.get("snapshot", {})
            status = snap.get("status", "no_log")
            live = self.mode == "live"
            source = snap.get("source", {})
            from tools.client_common import content_id, log_game_key
            from tools.client_actions import frame_identity
            game_key = log_game_key(source)
            if game_key != self.game_key:
                self.game_key, self.dismissed_frame = game_key, None
                record = self.race_records.get(game_key)
                self.lobby_races = record.get("races") if isinstance(record, dict) else record
                self.lobby_races_source = record.get("source") if isinstance(record, dict) else "user_confirmed_for_game"
                if live: self.proposal = None
            needs_races = live and bool(self.game_key) and not self.lobby_races and snap.get("game_status") == "running"
            race_status = None
            if self.race_reader:
                race_status = self.race_reader.sample(self.game_key, source,
                    enabled=needs_races and reading.get("running") is not False)
                if (needs_races and race_status.get("status") == "ready"
                        and race_status.get("game_key") == self.game_key):
                    self._remember_races(race_status["races"], "hearthmirror", race_status)
                    needs_races = False
            if self.lobby_races:
                snap = {**snap, "lobby_races": self.lobby_races, "lobby_races_source": self.lobby_races_source}
            memory_status, scene_source = None, "loading_screen_log"
            if self.memory_reader:
                from tools.client_memory import merge_public_state
                memory = self.memory_reader.sample(self.game_key, source,
                    enabled=live and reading.get("running") is not False)
                record = self.race_records.get(self.game_key)
                evidence = record.get("evidence") if isinstance(record, dict) else None
                snap, scene, memory_status = merge_public_state(snap, memory, evidence)
                if scene:
                    reading = {**reading, "scene": scene}
                    scene_source = "hearthmirror"
            combat = snap.get("game_status") != "complete" and (
                snap.get("client_phase") == "combat" or any(e.get("type") == "GAME" and
                    e.get("tags", {}).get("BACON_IN_COMBAT_PHASE") == 1 for e in snap.get("entities", [])))
            memory_blocks = bool(memory_status and memory_status.get("blocks_actions"))
            model_state = self.model_bridge.sample(snap, enabled=live and reading.get("running") is not False
                and not combat and not memory_blocks and snap.get("freshness", {}).get("status") == "recent") if self.model_bridge else None
            model_title, model_detail, model_blockers = model_summary(model_state)
            if needs_races and race_status:
                model_title = "正在读取本局种族"
                model_detail = race_status["detail"]
                model_blockers = [model_detail]
            if live and combat:
                model_title, model_detail, model_blockers = "正在战斗", "回到酒馆后自动更新操作建议。", []
            if live and memory_blocks:
                model_title, model_detail, model_blockers = "等待对局界面", "进入本局操作界面后自动更新建议。", []
            if live:
                key = frame_identity(snap)
                fresh = not combat and not memory_blocks and reading.get("running") is not False and snap.get("freshness", {}).get("status") == "recent" and snap.get("game_status") != "complete"
                if self.proposal and (not fresh or self.proposal["frame_id"] != key):
                    self.proposal = None
                    self.message = None
                decision = (model_state or {}).get("proposal")
                if fresh and decision and decision.get("frame_id") == key and self.dismissed_frame != key and not self.proposal:
                    b = decision["binding"]
                    self.proposal = {"id": content_id([key, decision["action"], decision["instruction"]]),
                        "frame_id": key, "scope": "client_live_advice", "title": TITLES.get(decision["kind"], "下一步操作"),
                        "instruction": decision["instruction"], "references": [],
                        "cost": {"amount": b["cost"], "currency": "health" if b.get("costs_health") else "gold"} if isinstance(b.get("cost"), int) else None,
                        "expires_at": self.clock()+self.ttl, "live_ready": True}
                    self.message = None
            log_session = Path(source["path"]).parent.name if source.get("path") else None
            titles = {"snapshot": "已读到日志快照", "incomplete_snapshot": "局面尚未稳定",
                      "empty_log": "等待对局局面", "missing_log": "等待对局局面",
                      "no_log": "等待客户端日志", "read_or_parse_error": "日志需要核对",
                      "no_complete_game_start": "等待完整对局记录"}
            cards, facts = [], []
            if live:
                cards, facts = project_live_cards(snap)
                facts.extend([{"label": "客户端", "value": reading.get("build") or "待确认"},
                         {"label": "当前界面" if scene_source == "hearthmirror" else "最近界面记录", "value": reading.get("scene") or "待核对"},
                         {"label": "日志会话", "value": log_session or "尚未发现"},
                         {"label": "日志大小", "value": f"{source.get('bytes', 0):,} 字节" if "bytes" in source else "尚未生成"}])
                if snap.get("performance"):
                    perf = snap["performance"]
                    first_read = perf.get("read_bytes", 0) > 0 and perf.get("read_bytes") == source.get("bytes")
                    facts.append({"label": "首次载入耗时" if first_read else "本次读取耗时", "value": f"{perf['total_ms']:.1f} 毫秒"})
                facts.extend([
                    {"label": "未知玩法卡牌", "value": str(len(snap.get("unknown_card_ids", [])))},
                    {"label": "内部实体／皮肤待映射", "value": str(len(snap.get("unmapped_internal_ids", [])) + len(snap.get("unmapped_skin_ids", [])))},
                    {"label": "日志兼容处理", "value": str(sum(snap.get("parser_diagnostics", {}).get("warning_counts", {}).values()))},
                    {"label": "已读取操作", "value": str(sum(snap.get("action_counts", {}).values()))}])
                facts.append({"label": "模型", "value": model_title})
                if model_state and model_state.get("input"):
                    coverage = model_state["input"]
                    facts.extend([
                        {"label": "模型特征已转换", "value": f"{coverage['known_features']} / {coverage['total_features']}"},
                        {"label": "卡牌槽位已确认", "value": f"{coverage['known_card_slots']} / {coverage['total_card_slots']}"},
                        {"label": "已映射可用动作", "value": str(len(coverage['mapped_actions']))}])
            elif self.frame:
                me = self.frame["me"]
                for zone in ("shop", "board", "hand"):
                    rows = me.get(zone, [])
                    if not isinstance(rows, list):
                        continue
                    for index, c in enumerate(rows):
                        if not isinstance(c, dict):
                            continue
                        name = c.get("name_zh") or c.get("name")
                        cards.append({"id": f"{zone}-{index}", "name": name if isinstance(name, str) else "未知卡牌",
                            "detail": f"{ZONE_LABELS[zone]}第 {index + 1} 位" + (f" · {c['atk']}/{c['hp']}" if 'atk' in c and 'hp' in c else ""),
                            "card_id": c.get("id") if isinstance(c.get("id"), str) else None})
                facts = [{"label": "回合", "value": str(self.frame.get("turn") or "—")},
                         {"label": "金币", "value": str(me.get("gold", "—"))},
                         {"label": "酒馆等级", "value": str(me.get("tier", "—"))}]
            operations = []
            if live:
                for event in reversed(snap.get("recent_actions", [])[-8:]):
                    kind = event["kind"]
                    ref = event.get("target" if kind in ("buy_minion", "buy_spell", "sell") else "main", {})
                    name = self.catalog.get(ref.get("card_id"), {}).get("name_zh")
                    label = OPERATION_LABELS.get(kind, "待分类操作")
                    operations.append({"id": str(event["sequence"]), "title": label + (" · " + name if name else ""),
                        "detail": event["at"].split(".")[0] + (" · 游戏已确认执行" if event.get("execution_verified") else " · 已读到客户端提交")})
            return {"schema": "autotavern.window.v1", "mode": self.mode,
                "revision": content_id((self.mode, self.game_key, self.lobby_races, snap.get("snapshot_id"), status, snap.get("freshness", {}).get("status"),
                                  source.get("path"), source.get("bytes"),
                                  reading.get("scene"), scene_source, snap.get("perspective"), reading.get("running"), self.message,
                                  model_title, model_detail, model_state, race_status and race_status.get("status"),
                                  self.proposal and self.proposal["id"], len(self.history))),
                "mode_label": {"live": "真实日志", "rehearsal": "操作演练", "replay": "模拟器回放"}[self.mode],
                "client_running": reading.get("running"), "client_build": reading.get("build"),
                "client_scene": reading.get("scene"),
                "client_scene_source": scene_source,
                "perspective": deepcopy(snap.get("perspective")),
                "memory_reader": memory_status,
                "client_phase": "combat" if combat else snap.get("client_phase"),
                "game_status": snap.get("game_status"),
                "recent_operations": operations,
                "state_title": ("本局已结束" if snap.get("game_status") == "complete" else titles.get(status, "等待局面")) if live else f"演练第 {self.index + 1} / 4 步" if self.mode == "rehearsal" else "模拟器动作预览",
                "state_detail": model_detail if live else "固定局面用于检查指令与反馈，不会操作游戏。",
                "model_status": {"title": model_title, "detail": model_detail},
                "game_key": self.game_key, "lobby_races": self.lobby_races,
                "lobby_races_source": self.lobby_races_source,
                "race_reader": race_status,
                "needs_lobby_races": bool(needs_races),
                "read_status": status, "freshness": snap.get("freshness", {}).get("status", "unknown"),
                "model_ready": bool((model_state or {}).get("policy")), "live_ready": bool(live and self.proposal),
                "proposal": deepcopy(self.proposal), "message": self.message,
                "can_next_frame": self.mode == "rehearsal" and self.proposal is None,
                "next_frame_label": "载入下一帧" if self.advance else "重新载入本帧",
                "cards": cards[:80], "facts": facts,
                "blockers": (model_blockers if model_state else ["尚未配置训练模型"]) if live else [],
                "history": deepcopy(self.history[:6]), "polled_at": self.clock()}
