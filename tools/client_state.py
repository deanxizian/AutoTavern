"""Read a player-visible Power.log snapshot without loading TavernGym or a model.

This is a diagnostic log bridge, not yet a validated Battlegrounds state/action
adapter. No account names, account IDs, raw log lines or inferred hidden cards
are exported. The game client and its configuration are never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import plistlib
import re
import time

from hearthstone.enums import BlockType, CardType, GameTag, Zone, State, Step
from hslog import LogParser, packets
from hslog.export import BaseExporter, FriendlyPlayerExporter
from hslog.parser import OptionsHandler
from hslog.player import coerce_to_entity_id

if __package__:
    from .client_common import (DEFAULT_MAX_AGE, content_id, freshness, load_catalog, newest_log,
                                write_private_json)
else:
    from client_common import (DEFAULT_MAX_AGE, content_id, freshness, load_catalog, newest_log,
                               write_private_json)


CREATE = re.compile(rb"GameState\.DebugPrintPower\(\)\s*-\s*CREATE_GAME\s*$")
DUPLICATE = re.compile(rb"^[DIEW] \d{2}:\d{2}:\d{2}\.\d+ PowerTaskList\.")
CLIENT_COMBAT = re.compile(rb"PowerTaskList\.DebugPrintPower\(\)\s*-\s*TAG_CHANGE Entity=GameEntity tag=BACON_IN_COMBAT_PHASE value=([01])\s*$")
CLIENT_COMPLETE = re.compile(rb"PowerTaskList\.DebugPrintPower\(\)\s*-\s*TAG_CHANGE Entity=GameEntity tag=STATE value=COMPLETE\s*$")
MAX_GAME_BYTES = 128 * 1024 * 1024


def client_phase_after(raw, previous):
    # GameState can contain the next recruit turn while combat is still playing.
    # Read only this presentation marker from PowerTaskList, not duplicate state.
    if b"BACON_IN_COMBAT_PHASE" in raw and (match := CLIENT_COMBAT.search(raw)):
        return "combat" if match[1] == b"1" else "recruit"
    if b"STATE value=COMPLETE" in raw and CLIENT_COMPLETE.search(raw):
        return "game_over"
    return previous


@lru_cache(maxsize=4096, typed=True)
def enum_name(value, cls=None):
    if cls is not None:
        try:
            value = cls(value)
        except ValueError:
            pass
    return getattr(value, "name", str(value))


def entity_id(value):
    return None if value is None or value == "" else int(coerce_to_entity_id(value))


class VisibleExporter(BaseExporter):
    """Replay only recorded entities/tags and preserve Options/Choices packets.

    The standard EntityTreeExporter omits options and choices, and its card
    helpers may consult a separate card database. This exporter does neither.
    """

    def __init__(self, tree):
        super().__init__(tree)
        self.entities = {}
        self.players = []
        self.options = None
        self.choices = None
        self.options_after_last_state_change = False
        self.choices_after_last_state_change = False
        self.reset_seen = False
        self.dirty = set()
        self.projected = {}
        self.entity_fingerprint = None
        self.actions = deque(maxlen=32)
        self.action_counts = Counter()
        self._server_actions = set()
        self._input_before = {}
        self._choice_events = {}
        self.choice_packet_complete = False
        self.options_packet_complete = False
        self.decision_changes = []
        # Event-time public history survives worker frame coalescing. It contains
        # only identities already revealed by this client's log.
        self.public_history = {"version": 1, "complete_from_start": False,
            "seen_boards": {}, "last_opponents": {}, "shop_at_combat": {},
            "powers": {}, "revealed": {}}
        self._blocks_seen = set()

    def changed(self):
        self.options_after_last_state_change = False
        self.choices_after_last_state_change = False

    def affects_options(self, row):
        """Leaderboard/opponent power updates do not replace our legal menu."""
        t = row.get("tags", {})
        local = next((p["player_id"] for p in self.players if 1 <= p["player_id"] <= 8), None)
        if local is None:
            return True
        if t.get(GameTag.CARDTYPE) in (CardType.HERO, CardType.HERO_POWER) and t.get(GameTag.CONTROLLER) != local:
            return False
        if t.get(GameTag.CARDTYPE) == CardType.HERO and t.get(GameTag.ZONE) == Zone.SETASIDE:
            return False
        return True

    def lookup(self, value):
        key = entity_id(value)
        if key not in self.entities:
            raise ValueError("Log references an entity that was not created")
        return self.entities[key]

    def handle_create_game(self, packet):
        self.public_history = {"version": 1, "complete_from_start": True,
            "seen_boards": {}, "last_opponents": {}, "shop_at_combat": {},
            "powers": {}, "revealed": {}}
        self._blocks_seen.clear()
        self._server_actions.clear()
        self._input_before.clear()
        self.entities.clear()
        self.players.clear()
        self.options = self.choices = None
        self.projected.clear()
        self.dirty.clear()
        self.changed()
        self.entities[entity_id(packet.entity)] = {"card_id": None, "tags": dict(packet.tags)}
        self.dirty.add(entity_id(packet.entity))
        for player in packet.players:
            self.handle_player(player)

    def handle_player(self, packet):
        key = entity_id(packet.entity)
        self.players.append({"entity_id": key, "player_id": packet.player_id})
        self.entities[key] = {"card_id": None, "tags": dict(packet.tags)}
        self.dirty.add(key)

    def handle_full_entity(self, packet):
        row = {"card_id": packet.card_id or None, "tags": dict(packet.tags)}
        if self.affects_options(row): self.changed()
        self.entities[entity_id(packet.entity)] = row
        self.dirty.add(entity_id(packet.entity))

    def handle_show_entity(self, packet):
        row = self.lookup(packet.entity)
        row["card_id"] = packet.card_id or None
        row["tags"].update(packet.tags)
        if self.affects_options(row): self.changed()
        self.dirty.add(entity_id(packet.entity))

    def handle_change_entity(self, packet):
        self.handle_show_entity(packet)

    def handle_hide_entity(self, packet):
        row = self.lookup(packet.entity)
        if self.affects_options(row): self.changed()
        if row.get("card_id"):
            self.public_history["revealed"][str(entity_id(packet.entity))] = row["card_id"]
        # A previously revealed identity is not presented as a currently visible card.
        row["card_id"] = None
        row["tags"][GameTag.ZONE] = packet.zone
        self.dirty.add(entity_id(packet.entity))

    def handle_tag_change(self, packet):
        row = self.lookup(packet.entity)
        previous = row["tags"].get(packet.tag)
        if previous != packet.value and (
            (packet.tag == GameTag.STEP and packet.value == Step.MAIN_END)
            or (packet.tag == GameTag.BACON_IN_COMBAT_PHASE and packet.value == 1)
        ):
            self.finish_input_window()
        if previous != packet.value:
            if self.affects_options(row): self.changed()
            self.decision_changes.append(enum_name(packet.tag, GameTag))
            self.decision_changes = self.decision_changes[-32:]
        row["tags"][packet.tag] = packet.value
        self.dirty.add(entity_id(packet.entity))
        if packet.tag == GameTag.BACON_IN_COMBAT_PHASE and packet.value == 1 and previous != 1:
            for p in self.players:
                owner = p["player_id"]
                hero = self.entities.get(self.entities[p["entity_id"]]["tags"].get(GameTag.HERO_ENTITY), {})
                if hero.get("card_id") == "TB_BaconShopBob":
                    self.public_history["shop_at_combat"][str(owner)] = [eid for eid, e in self.entities.items()
                        if e["tags"].get(GameTag.CONTROLLER) == owner and e["tags"].get(GameTag.ZONE) == Zone.PLAY
                        and e["tags"].get(GameTag.CARDTYPE) in (CardType.MINION, CardType.BATTLEGROUND_SPELL)]
        # Client combat-setup boundary, before start-of-combat effects and attacks.
        if int(packet.tag) == 2022 and previous == 1 and packet.value == 0:
            self.remember_combat()

    def handle_reset_game(self, packet):
        self.changed()
        self.reset_seen = True
        self.public_history["complete_from_start"] = False
        self.options = self.choices = None
        for row in self.entities.values():
            row["card_id"] = None
        self.dirty.update(self.entities)

    def handle_block(self, packet):
        self.begin_block(packet)
        super().handle_block(packet)

    def begin_block(self, packet):
        if packet.type == BlockType.GAME_RESET:
            self.handle_reset_game(packet)
        if packet.type != BlockType.POWER or id(packet) in self._blocks_seen:
            return
        self._blocks_seen.add(id(packet))
        row = self.entities.get(entity_id(packet.entity), {})
        cid, tags = row.get("card_id"), row.get("tags", {})
        if cid:
            owner = str(tags.get(GameTag.CONTROLLER))
            counts = self.public_history["powers"].setdefault(owner, {})
            counts[cid] = counts.get(cid, 0) + 1
        if self.actions:
            event = self.actions[-1]
            if event.get("main", {}).get("entity_id") == entity_id(packet.entity):
                self._server_actions.add(event["sequence"])

    def confirm_actions(self):
        """Server evidence within an input's response window confirms it.

        A later retry or combat must not lend its transitions to an old input.
        Merely emitting SendOption never advances decoding history.
        """
        for event in self.actions:
            seq = event["sequence"]
            if event.get("execution_verified") or event["kind"] == "choice": continue
            main, target = event.get("main", {}), event.get("target", {})
            kind = event["kind"]
            current = self.reference(main.get("entity_id"))
            target_now = self.reference(target.get("entity_id"))
            before = self._input_before.get(seq, {})
            if before.get("closed"):
                continue
            proof = seq in self._server_actions
            if kind in ("play_minion", "cast_spell"):
                proof |= main.get("zone") == "HAND" and current.get("zone") != "HAND"
            elif kind.startswith("buy_"):
                proof |= target.get("zone") == "PLAY" and target_now.get("zone") in ("HAND", "GRAVEYARD")
            elif kind == "sell":
                proof |= target.get("zone") == "PLAY" and target_now.get("zone") != "PLAY"
            elif kind in ("board_position", "shop_position"):
                proof = event.get("position") != main.get("position") and current.get("position") == event.get("position")
            elif kind == "upgrade":
                p = self.entities.get(before.get("player"), {}).get("tags", {})
                proof |= p.get(GameTag.PLAYER_TECH_LEVEL, 1) > before.get("tier", 1)
            elif kind == "freeze_toggle":
                old = before.get("frozen", {})
                proof |= bool(old) and all(self.entities.get(eid, {}).get("tags", {}).get(GameTag.FROZEN, 0) != value for eid, value in old.items())
            if not proof: continue
            event["execution_verified"] = True
            event["evidence"] = "server_power_or_entity_transition_within_input_window"
            confirmations = self.public_history.setdefault("confirmed_actions", [])
            confirmations.append({"sequence": seq, "kind": kind, "entity_id": main.get("entity_id"),
                "target_entity_id": target.get("entity_id"), "position": event.get("position"),
                "execution_verified": True, "evidence": event["evidence"]})
            del confirmations[:-32]
            if main.get("controller"):
                stats = self.public_history.setdefault("stats", {}).setdefault(str(main["controller"]), {})
                keys = {"buy_minion": ("minions_bought", "cards_bought"), "buy_spell": ("cards_bought",),
                    "sell": ("minions_sold",), "refresh": ("refreshes",),
                    "play_minion": ("cards_played",), "cast_spell": ("cards_played",)}.get(kind, ())
                for key in keys: stats[key] = stats.get(key, 0)+1

    def finish_input_window(self):
        # Preserve any response already received before closing the window. This
        # also handles rapid inputs whose next menu has not yet been exported.
        self.confirm_actions()
        for event in self.actions:
            before = self._input_before.get(event["sequence"])
            if before is not None and not event.get("execution_verified"):
                before["closed"] = True

    def remember_combat(self):
        from copy import deepcopy
        for p in self.players:
            tags = self.entities[p["entity_id"]]["tags"]
            combat_id = tags.get(GameTag.BACON_CURRENT_COMBAT_PLAYER_ID, 0)
            owner = tags.get(GameTag.CONTROLLER, p["player_id"])
            if not combat_id:
                continue
            board_ids = [eid for eid, e in self.entities.items() if e["card_id"] and
                e["tags"].get(GameTag.CONTROLLER) == owner and e["tags"].get(GameTag.ZONE) == Zone.PLAY
                and e["tags"].get(GameTag.CARDTYPE) == CardType.MINION]
            board_ids.sort(key=lambda eid: self.entities[eid]["tags"].get(GameTag.ZONE_POSITION, 0))
            ids = set(board_ids)
            ids.update(eid for eid, e in self.entities.items() if e["card_id"] and
                e["tags"].get(GameTag.ATTACHED) in ids and e["tags"].get(GameTag.ZONE) == Zone.PLAY)
            rows = [{"entity_id": eid, "card_id": self.entities[eid]["card_id"],
                "tags": {enum_name(k, GameTag): int(v) for k, v in self.entities[eid]["tags"].items()}}
                for eid in sorted(ids)]
            self.public_history["seen_boards"][str(combat_id)] = {"board_ids": board_ids,
                "entities": deepcopy(rows), "turn": tags.get(GameTag.TURN, 0)}
            for other in self.players:
                if other["entity_id"] != p["entity_id"]:
                    self.public_history["last_opponents"][str(other["player_id"])] = combat_id

    @staticmethod
    def option(packet):
        return {"id": packet.id, "entity_id": entity_id(packet.entity),
                "type": enum_name(packet.type), "kind": packet.optype,
                "error": enum_name(packet.error) if packet.error is not None else None,
                "children": [VisibleExporter.option(p) for p in packet.options]}

    def handle_options(self, packet):
        self.confirm_actions()
        if self.actions:
            event = self.actions[-1]
            if event["kind"] == "board_position" and event.get("execution_verified") and event.get("position") != event["main"].get("position"):
                row = self.entities.get(event["main"]["entity_id"], {})
                if row.get("tags", {}).get(GameTag.ZONE_POSITION) == event["position"]:
                    p = next((x for x in self.players if x["player_id"] == event["main"]["controller"]), None)
                    if p:
                        self.public_history["confirmed_move"] = {"sequence": event["sequence"],
                            "turn": self.entities[p["entity_id"]]["tags"].get(GameTag.TURN),
                            "entity_id": event["main"]["entity_id"], "position": event["position"]}
        self.options = {"id": packet.id, "options": [self.option(p) for p in packet.options]}
        self.options_after_last_state_change = True
        self.options_packet_complete = True
        self.decision_changes = []

    def handle_send_option(self, packet):
        self.finish_input_window()
        option = next((o for o in (self.options or {}).get("options", [])
                       if o["id"] == packet.option), {})
        main = self.reference(option.get("entity_id"))
        target = self.reference(entity_id(packet.target))
        cid = main.get("card_id") or ""
        kind = "unclassified"
        if cid.startswith("TB_BaconShop_DragBuy"):
            kind = "buy_spell" if "Spell" in cid else "buy_minion"
        elif cid == "TB_BaconShop_DragSell": kind = "sell"
        elif cid == "TB_BaconShop_8p_Reroll_Button": kind = "refresh"
        elif cid.startswith("TB_BaconShopTechUp"): kind = "upgrade"
        elif cid == "TB_BaconShopLockAll_Button": kind = "freeze_toggle"
        elif cid == "BG36_Button_DarkGift": kind = "dark_gift"
        elif main["zone"] == "HAND":
            kind = "play_minion" if main["type"] == "MINION" else "cast_spell"
        elif main["type"] == "HERO_POWER": kind = "hero_power"
        elif main["type"] in ("MINION", "BATTLEGROUND_SPELL") and main["zone"] == "PLAY":
            bartender = any(e["card_id"] == "TB_BaconShopBob" and
                             e["tags"].get(GameTag.ZONE) == Zone.PLAY and
                             e["tags"].get(GameTag.CONTROLLER) == main["controller"]
                             for e in self.entities.values())
            if packet.position > 0:
                kind = "shop_position" if bartender else "board_position"
            elif main["activate_available"]:
                kind = "activate_minion"
        self.action_counts[kind] += 1
        self.actions.append({"sequence": sum(self.action_counts.values()), "at": str(packet.ts),
            "kind": kind, "evidence": "client_input_not_execution_confirmation",
            "options_id": (self.options or {}).get("id"), "option": packet.option,
            "main": main, "target": target, "position": packet.position,
            "suboption": packet.suboption})
        player = next((p["entity_id"] for p in self.players if p["player_id"] == main.get("controller")), None)
        self._input_before[sum(self.action_counts.values())] = {"player": player,
            "tier": self.entities.get(player, {}).get("tags", {}).get(GameTag.PLAYER_TECH_LEVEL, 1),
            "frozen": {eid: e["tags"].get(GameTag.FROZEN, 0) for eid, e in self.entities.items()
                if e["tags"].get(GameTag.ZONE) == Zone.PLAY and e["tags"].get(GameTag.CARDTYPE) == CardType.MINION and e["tags"].get(GameTag.CONTROLLER) != main.get("controller")}}
        self._input_before = {s: v for s, v in self._input_before.items() if s > sum(self.action_counts.values())-32}
        self.changed()

    def reference(self, key):
        row = self.entities.get(key, {})
        tags = row.get("tags", {})
        return {"entity_id": key, "card_id": row.get("card_id"),
                "controller": tags.get(GameTag.CONTROLLER),
                "zone": enum_name(tags.get(GameTag.ZONE, 0), Zone),
                "type": enum_name(tags.get(GameTag.CARDTYPE, 0), CardType),
                "position": tags.get(GameTag.ZONE_POSITION), "cost": tags.get(GameTag.COST),
                "activate_available": bool(tags.get(GameTag.INTERACTABLE_OBJECT)),
                "activate_cost": tags.get(GameTag.INTERACTABLE_OBJECT_COST)}

    def render_entities(self, cards):
        changed = bool(self.dirty)
        for key in self.dirty:
            entity = self.entities[key]
            tags, card_id = entity["tags"], entity["card_id"]
            if card_id:
                self.public_history["revealed"][str(key)] = card_id
            card = cards.get(card_id, {})
            kind = enum_name(tags.get(GameTag.CARDTYPE, 0), CardType)
            category = ("known_card" if card else "hidden" if not card_id else
                        "internal" if kind in ("ENCHANTMENT", "GAME_MODE_BUTTON", "MOVE_MINION_HOVER_TARGET") else
                        "hero_skin" if kind == "HERO" and tags.get(GameTag.BACON_SKIN) else "unknown_card")
            self.projected[key] = {"entity_id": key, "card_id": card_id,
                "catalog_name": card.get("name"), "catalog_name_zh": card.get("name_zh"),
                "controller": tags.get(GameTag.CONTROLLER),
                "zone": enum_name(tags.get(GameTag.ZONE, 0), Zone),
                "position": tags.get(GameTag.ZONE_POSITION), "type": kind,
                "catalog_category": category,
                "tags": {enum_name(k, GameTag): int(v) for k, v in tags.items()}}
        self.dirty.clear()
        rows = [self.projected[k] for k in sorted(self.entities)]
        if changed or self.entity_fingerprint is None:
            self.entity_fingerprint = content_id(rows)
        return rows

    def handle_choices(self, packet):
        self.confirm_actions()
        self.choices = {"id": packet.id, "type": enum_name(packet.type),
                        "player_entity_id": entity_id(packet.entity),
                        "source_entity_id": entity_id(packet.source),
                        "min": packet.min, "max": packet.max,
                        "entity_ids": [entity_id(p) for p in packet.choices],
                        "selection_sent_or_resolved": False}
        self.changed()
        self.choices_after_last_state_change = True
        self.choice_packet_complete = True
        self.decision_changes = []

    def handle_send_choices(self, packet):
        if type(packet) is packets.SendChoices:
            self.finish_input_window()
            key = (packet.id, str(packet.ts))
            if key not in self._choice_events:
                self.action_counts["choice"] += 1
                self._choice_events[key] = {"sequence": sum(self.action_counts.values()),
                    "at": str(packet.ts), "kind": "choice", "choices_id": packet.id,
                    "evidence": "client_input_not_execution_confirmation"}
                self.actions.append(self._choice_events[key])
            self._choice_events[key]["entity_ids"] = [entity_id(v) for v in packet.choices]
        if self.choices is not None and self.choices["id"] == packet.id:
            self.choices["selection_sent_or_resolved"] = True
        self.changed()

    def handle_chosen_entities(self, packet):
        self.handle_send_choices(packet)
        for event in reversed(self.actions):
            if event.get("choices_id") == packet.id and event.get("entity_ids") == [entity_id(v) for v in packet.choices]:
                event["execution_verified"] = True
                event["evidence"] = "server_chosen_entities"
                break


class DiagnosticOptions(OptionsHandler):
    """The pinned hslog workaround, recorded once rather than logging on every replay."""
    def __init__(self, parser):
        super().__init__()
        self.parser = parser

    def _check_for_options_hack(self, state, ts):
        while isinstance(state.current_block, packets.Block) and state.current_block.parent:
            self.parser.warning_counts["battlegrounds_missing_block_end"] += 1
            self.parser.recent_warnings.append({"code": "battlegrounds_missing_block_end", "at": str(ts)})
            state.block_end(ts)


class DiagnosticParser(LogParser):
    def __init__(self):
        super().__init__()
        self.warning_counts = Counter()
        self.recent_warnings = deque(maxlen=16)
        self._options_handler = DiagnosticOptions(self)


def last_game_lines(path):
    """Bound reading to the initial file size; never consume a partial trailing line."""
    rows, total, partial = [], 0, False
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        data = stream.read(before.st_size)
        for line_number, raw in enumerate(data.splitlines(keepends=True), 1):
            if not raw.endswith(b"\n"):
                partial = True
                break
            created = b"CREATE_GAME" in raw and CREATE.search(raw)
            if created:
                rows, total = [], 0
            if rows or created:
                rows.append((line_number, raw))
                total += len(raw)
                if total > MAX_GAME_BYTES:
                    raise ValueError("Current game exceeds the log reader size limit")
        after = os.fstat(stream.fileno())
    try:
        current = path.stat()
    except FileNotFoundError:
        current = None
    changed = current is None or file_signature(before) != file_signature(after) \
        or file_signature(before) != file_signature(current)
    return rows, partial, changed, before


def file_signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def open_blocks(tree):
    count = 0
    for packet in tree:
        if isinstance(packet, (packets.Block, packets.SubSpell)):
            count += int(not packet.ended) + open_blocks(packet.packets)
    return count


def validate_snapshot(result, *, now=None, max_age=DEFAULT_MAX_AGE):
    old_codes = {i["code"] for i in result.get("validation", {}).get("issues", [])}
    result["model_blockers"] = [b for b in result["model_blockers"] if b not in old_codes]
    source = result.get("source", {})
    result["freshness"] = freshness(source.get("modified_at"), now=now, max_age=max_age)
    result["freshness"]["basis"] = "log_mtime_not_live_client"
    if result["freshness"]["age_seconds"] is not None:
        source["age_seconds"] = max(0, result["freshness"]["age_seconds"])
    issues = []
    def issue(code, message):
        issues.append({"code": code, "message": message})
    if result["status"] not in ("snapshot", "incomplete_snapshot"):
        issue(result["status"], "尚未读到完整对局；保留记录，等待获准联调时再核对日志。")
    if result["freshness"]["status"] != "recent":
        issue("log_" + result["freshness"]["status"], "日志时间未知、过期或异常，不能作为当前局面。")
    if result["status"] == "incomplete_snapshot":
        issue("incomplete_log", "日志仍在写入、发生替换、重连或存在未结束事件。")
    if result.get("unknown_card_ids"):
        issue("unknown_cards", "存在词表外卡牌；需核对实际卡牌 ID，不得替换为相似卡牌。")
    if result.get("unmapped_internal_ids") or result.get("unmapped_skin_ids"):
        issue("internal_entity_mapping_unvalidated", "按钮、附魔或皮肤已单独分类，其效果映射仍需验证。")
    if result.get("perspective", {}).get("source") != "explicit_player_id":
        issue("perspective_unconfirmed", "本方玩家身份尚未人工核对。")
    if result.get("last_choices") and not result["last_choices"]["selection_sent_or_resolved"]:
        if not result.get("choices_after_last_state_change"):
            issue("stale_choices", "选择包之后局面已变化；选项顺序和可用性需要重新核对。")
    elif not result.get("options_after_last_state_change"):
        issue("missing_current_options", "缺少当前有效的操作包，不能推导合法动作。")
    issue("log_build_unverified", "已安装客户端版本不能证明这份日志的版本或服务端规则。")
    issue("legal_action_mapping_unvalidated", "日志操作包尚未转换为经验证的模型合法动作。")
    result["validation"] = {"status": "blocked", "issues": issues}
    result["model_blockers"] = list(dict.fromkeys(
        result["model_blockers"] + [i["code"] for i in issues]))
    return result


@lru_cache(maxsize=1)
def parser_versions():
    return {p: version(p) for p in ("hslog", "hearthstone")}


def base_snapshot(meta, installed_build=None, now=None):
    result = {"schema": "autotavern.power-log.v1", "status": "no_log",
              "captured_at": (now or datetime.now(timezone.utc)).isoformat(),
              "model_ready": False, "live_ready": False,
              "model_blockers": ["live_state_and_action_mapping_unvalidated"],
              "parser": dict(parser_versions())}
    result["versions"] = {"simulator_client": meta["client_version"],
                          "simulator_server_rules": meta["server_patch"],
                          "installed_client": installed_build, "log_build_verified": False}
    if installed_build and installed_build != meta["client_version"]:
        result["model_blockers"].append("installed_client_differs_from_simulator")
    return result


def populate_snapshot(result, exported, parser, cards, player_id, perspective_source, block_count):
    if player_id is not None and player_id not in {p["player_id"] for p in exported.players}:
        raise ValueError("Requested player is absent from this log")
    result["perspective"] = {"player_id": player_id, "source": perspective_source}
    source = result["source"]
    result["snapshot_id"] = content_id({
        **{k: source[k] for k in ("path", "device", "inode", "game_start_line", "parsed_game_sha256")},
        "player_id": player_id, "perspective_source": perspective_source,
    })
    result["players"] = list(exported.players)
    result["entities"] = exported.render_entities(cards)
    from copy import deepcopy
    result["public_history"] = deepcopy(exported.public_history)
    result["model_snapshot_id"] = content_id({"entities": exported.entity_fingerprint,
        "public_history": result["public_history"], "player_id": player_id})
    def identities(category):
        return sorted({e["card_id"] for e in result["entities"] if e["catalog_category"] == category})
    result["unknown_card_ids"] = identities("unknown_card")
    result["unmapped_internal_ids"] = identities("internal")
    result["unmapped_skin_ids"] = identities("hero_skin")
    result["last_options"] = exported.options
    # Copies protect prior snapshots when a partial choice packet grows on the next read.
    result["last_choices"] = dict(exported.choices) if exported.choices else None
    result["options_after_last_state_change"] = exported.options_after_last_state_change
    result["choices_after_last_state_change"] = exported.choices_after_last_state_change
    result["choice_packet_complete"] = exported.choice_packet_complete
    result["options_packet_complete"] = exported.options_packet_complete
    result["decision_changes"] = list(exported.decision_changes)
    result["recent_actions"] = [dict(a) for a in exported.actions]
    result["action_counts"] = dict(exported.action_counts)
    result["parser_diagnostics"] = {"warning_counts": dict(parser.warning_counts),
                                    "recent_warnings": list(parser.recent_warnings)}
    result["open_blocks"] = block_count
    result["reset_seen"] = exported.reset_seen
    game = next((e for e in result["entities"] if e["type"] == "GAME"), None)
    result["game_status"] = "complete" if game and game["tags"].get("STATE") == State.COMPLETE else "running"
    result["status"] = ("incomplete_snapshot" if source["partial_trailing_line"] or
                        source["changed_while_reading"] or block_count or exported.reset_seen else "snapshot")


def snapshot(path, *, player_id=None, installed_build=None, max_age=DEFAULT_MAX_AGE, now=None):
    meta, cards = load_catalog()
    result = base_snapshot(meta, installed_build, now)
    if path is None:
        return validate_snapshot(result, now=now, max_age=max_age)
    result["source"] = {"path": str(path.resolve())}
    line_number = None
    try:
        rows, partial, changed, stat = last_game_lines(path)
        result["source"].update({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                 "device": stat.st_dev, "inode": stat.st_ino,
                                 "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                                 "age_seconds": max(0, time.time() - stat.st_mtime),
                                 "changed_while_reading": changed,
                                 "partial_trailing_line": partial})
        if not rows:
            result["status"] = "empty_log" if stat.st_size == 0 else "no_complete_game_start"
            return validate_snapshot(result, now=now, max_age=max_age)
        result["source"]["game_start_line"] = rows[0][0]
        result["source"]["last_complete_line"] = rows[-1][0]
        result["source"]["parsed_game_sha256"] = hashlib.sha256(b"".join(r for _, r in rows)).hexdigest()
        parser = DiagnosticParser()
        client_phase = None
        for line_number, raw in rows:
            client_phase = client_phase_after(raw, client_phase)
            if raw.strip() and not DUPLICATE.match(raw):
                parser.read_line(raw.decode("utf-8"))
        parser.flush()
        tree = parser.games[-1]
        exported = VisibleExporter(tree).export()
        if player_id is None:
            player_id = FriendlyPlayerExporter(tree).export()
            perspective_source = "hslog_inference_unvalidated_for_this_game"
        else:
            perspective_source = "explicit_player_id"
        populate_snapshot(result, exported, parser, cards, player_id, perspective_source, open_blocks(tree))
        result["client_phase"] = client_phase
    except FileNotFoundError:
        result["status"] = "missing_log"
    except Exception as error:
        # Parser exception messages can contain full log lines and player names.
        result["status"] = "read_or_parse_error"
        result["error"] = {"type": type(error).__name__, "line": line_number}
    return validate_snapshot(result, now=now, max_age=max_age)


def write_snapshot(path, value):
    write_private_json(path, value)


def watch_signature(path, *, max_age=DEFAULT_MAX_AGE, now=None):
    """Refresh on writes/rotation AND age transitions, even when a log is idle."""
    if path is None:
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return str(path), "missing"
    observed = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
    return str(path), file_signature(stat), freshness(observed, now=now, max_age=max_age)["status"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path, help="Read one Power.log")
    source.add_argument("--logs", type=Path, help="Select the newest Power.log in a Logs directory")
    parser.add_argument("--client-app", type=Path, help="Read the installed macOS app build from Info.plist")
    parser.add_argument("--player-id", type=int, help="Override the unvalidated friendly-player inference")
    parser.add_argument("--out", type=Path, help="Atomically replace a local JSON snapshot")
    parser.add_argument("--watch", action="store_true", help="Refresh when the log changes or rotates")
    parser.add_argument("--interval", type=float, default=.05)
    parser.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE,
                        help="Mark logs older than this many seconds as stale (default: 30)")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < .02:
        parser.error("--interval must be at least 0.02 seconds")
    if not math.isfinite(args.max_age) or args.max_age <= 0:
        parser.error("--max-age must be finite and positive")
    build = None
    if args.client_app:
        try:
            with (args.client_app / "Contents/Info.plist").open("rb") as stream:
                info = plistlib.load(stream)
                build = info.get("CFBundleVersion") if isinstance(info, dict) else None
                if not isinstance(build, str) or not build:
                    raise ValueError("Missing build")
        except (OSError, ValueError, plistlib.InvalidFileException):
            parser.error("Cannot read client Info.plist; use client_audit.py for installation diagnostics")
    previous = object()
    if __package__:
        from .client_stream import IncrementalReader
    else:
        from client_stream import IncrementalReader
    reader = IncrementalReader(player_id=args.player_id, installed_build=build, max_age=args.max_age)
    result = {}
    try:
        while True:
            path = args.log if args.log is not None else newest_log(args.logs)
            if args.out and path and args.out.resolve() == path.resolve():
                parser.error("--out must not overwrite the source Power.log")
            signature = watch_signature(path, max_age=args.max_age)
            choices = result.get("last_choices") or {}
            settling = (result.get("last_options") and result.get("options_packet_complete") is False) or (
                choices and not choices.get("selection_sent_or_resolved") and result.get("choice_packet_complete") is False)
            if signature != previous or settling:
                result = reader.read(path)
                if args.out:
                    write_snapshot(args.out, result)
                    print(json.dumps({"status": result["status"], "out": str(args.out)}), flush=True)
                else:
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                previous = object() if result.get("source", {}).get("changed_while_reading") else signature
            if not args.watch:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
