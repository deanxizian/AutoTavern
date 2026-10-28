"""Read a player-visible Power.log snapshot without loading TavernGym or a model.

This is a diagnostic log bridge, not yet a validated Battlegrounds state/action
adapter. No account names, account IDs, raw log lines or inferred hidden cards
are exported. The game client and its configuration are never modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import plistlib
import re
import time

from hearthstone.enums import BlockType, CardType, GameTag, Zone
from hslog import LogParser, packets
from hslog.export import BaseExporter, FriendlyPlayerExporter
from hslog.player import coerce_to_entity_id


CREATE = re.compile(rb"GameState\.DebugPrintPower\(\)\s*-\s*CREATE_GAME\s*$")
MAX_GAME_BYTES = 64 * 1024 * 1024
CATALOG = Path(__file__).resolve().parents[1] / "crates/taverngym/data/cards.json"


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
        self.reset_seen = False

    def changed(self):
        self.options_after_last_state_change = False

    def lookup(self, value):
        key = entity_id(value)
        if key not in self.entities:
            raise ValueError("Log references an entity that was not created")
        return self.entities[key]

    def handle_create_game(self, packet):
        self.entities.clear()
        self.players.clear()
        self.options = self.choices = None
        self.changed()
        self.entities[entity_id(packet.entity)] = {"card_id": None, "tags": dict(packet.tags)}
        for player in packet.players:
            self.handle_player(player)

    def handle_player(self, packet):
        key = entity_id(packet.entity)
        self.players.append({"entity_id": key, "player_id": packet.player_id})
        self.entities[key] = {"card_id": None, "tags": dict(packet.tags)}

    def handle_full_entity(self, packet):
        self.changed()
        self.entities[entity_id(packet.entity)] = {
            "card_id": packet.card_id or None, "tags": dict(packet.tags)}

    def handle_show_entity(self, packet):
        self.changed()
        row = self.lookup(packet.entity)
        row["card_id"] = packet.card_id or None
        row["tags"].update(packet.tags)

    def handle_change_entity(self, packet):
        self.handle_show_entity(packet)

    def handle_hide_entity(self, packet):
        self.changed()
        row = self.lookup(packet.entity)
        # A previously revealed identity is not presented as a currently visible card.
        row["card_id"] = None
        row["tags"][GameTag.ZONE] = packet.zone

    def handle_tag_change(self, packet):
        self.changed()
        self.lookup(packet.entity)["tags"][packet.tag] = packet.value

    def handle_reset_game(self, packet):
        self.changed()
        self.reset_seen = True
        self.options = self.choices = None
        for row in self.entities.values():
            row["card_id"] = None

    def handle_block(self, packet):
        if packet.type == BlockType.GAME_RESET:
            self.handle_reset_game(packet)
        super().handle_block(packet)

    @staticmethod
    def option(packet):
        return {"id": packet.id, "entity_id": entity_id(packet.entity),
                "type": enum_name(packet.type), "kind": packet.optype,
                "error": enum_name(packet.error) if packet.error is not None else None,
                "children": [VisibleExporter.option(p) for p in packet.options]}

    def handle_options(self, packet):
        self.options = {"id": packet.id, "options": [self.option(p) for p in packet.options]}
        self.options_after_last_state_change = True

    def handle_send_option(self, packet):
        self.changed()

    def handle_choices(self, packet):
        self.choices = {"id": packet.id, "type": enum_name(packet.type),
                        "player_entity_id": entity_id(packet.entity),
                        "source_entity_id": entity_id(packet.source),
                        "min": packet.min, "max": packet.max,
                        "entity_ids": [entity_id(p) for p in packet.choices],
                        "selection_sent_or_resolved": False}
        self.changed()

    def handle_send_choices(self, packet):
        if self.choices is not None and self.choices["id"] == packet.id:
            self.choices["selection_sent_or_resolved"] = True
        self.changed()

    def handle_chosen_entities(self, packet):
        self.handle_send_choices(packet)


def newest_log(directory):
    candidates = [p for p in directory.glob("*/Power.log") if p.is_file()]
    if (directory / "Power.log").is_file():
        candidates.append(directory / "Power.log")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)), default=None)


def last_game_lines(path):
    """Bound reading to the initial file size; never consume a partial trailing line."""
    rows, total, partial = [], 0, False
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        line_number = 0
        while stream.tell() < before.st_size:
            raw = stream.readline(before.st_size - stream.tell())
            line_number += 1
            if not raw.endswith(b"\n"):
                partial = True
                break
            if CREATE.search(raw):
                rows, total = [], 0
            if rows or CREATE.search(raw):
                rows.append((line_number, raw))
                total += len(raw)
                if total > MAX_GAME_BYTES:
                    raise ValueError("Current game exceeds the log reader size limit")
        after = os.fstat(stream.fileno())
    changed = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    return rows, partial, changed, before


def open_blocks(tree):
    count = 0
    for packet in tree:
        if isinstance(packet, (packets.Block, packets.SubSpell)):
            count += int(not packet.ended) + open_blocks(packet.packets)
    return count


def snapshot(path, *, player_id=None, installed_build=None):
    result = {"schema": "autotavern.power-log.v1", "status": "no_log",
              "captured_at": datetime.now(timezone.utc).isoformat(),
              "model_ready": False, "model_blockers": ["live_state_and_action_mapping_unvalidated"],
              "parser": {p: version(p) for p in ("hslog", "hearthstone")}}
    catalog = json.loads(CATALOG.read_text())
    cards = {c["id"]: c for c in catalog["cards"]}
    result["versions"] = {"simulator_client": catalog["meta"]["client_version"],
                          "simulator_server_rules": catalog["meta"]["server_patch"],
                          "installed_client": installed_build, "log_build_verified": False}
    if installed_build and installed_build != catalog["meta"]["client_version"]:
        result["model_blockers"].append("installed_client_differs_from_simulator")
    if path is None:
        return result
    result["source"] = {"path": str(path.resolve())}
    line_number = None
    try:
        rows, partial, changed, stat = last_game_lines(path)
        result["source"].update({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                 "age_seconds": max(0, time.time() - stat.st_mtime),
                                 "changed_while_reading": changed,
                                 "partial_trailing_line": partial})
        if not rows:
            result["status"] = "empty_log" if stat.st_size == 0 else "no_complete_game_start"
            return result
        result["source"]["game_start_line"] = rows[0][0]
        result["source"]["last_complete_line"] = rows[-1][0]
        result["source"]["parsed_game_sha256"] = hashlib.sha256(b"".join(r for _, r in rows)).hexdigest()
        parser = LogParser()
        for line_number, raw in rows:
            if raw.strip():
                parser.read_line(raw.decode("utf-8"))
        parser.flush()
        tree = parser.games[-1]
        exported = VisibleExporter(tree).export()
        if player_id is None:
            player_id = FriendlyPlayerExporter(tree).export()
            perspective_source = "hslog_inference_unvalidated_for_this_game"
        else:
            perspective_source = "explicit_player_id"
        if player_id is not None and player_id not in {p["player_id"] for p in exported.players}:
            raise ValueError("Requested player is absent from this log")
        result["perspective"] = {"player_id": player_id, "source": perspective_source}
        result["players"] = exported.players
        result["entities"] = []
        for key, entity in sorted(exported.entities.items()):
            tags, card_id = entity["tags"], entity["card_id"]
            card = cards.get(card_id, {})
            result["entities"].append({
                "entity_id": key, "card_id": card_id,
                "catalog_name": card.get("name"), "catalog_name_zh": card.get("name_zh"),
                "controller": tags.get(GameTag.CONTROLLER),
                "zone": enum_name(tags.get(GameTag.ZONE, 0), Zone),
                "position": tags.get(GameTag.ZONE_POSITION),
                "type": enum_name(tags.get(GameTag.CARDTYPE, 0), CardType),
                "tags": {enum_name(k, GameTag): int(v) for k, v in tags.items()}})
        result["unknown_card_ids"] = sorted({e["card_id"] for e in result["entities"]
                                             if e["card_id"] and e["card_id"] not in cards})
        result["last_options"] = exported.options
        result["options_after_last_state_change"] = exported.options_after_last_state_change
        result["last_choices"] = exported.choices
        result["open_blocks"] = open_blocks(tree)
        result["reset_seen"] = exported.reset_seen
        result["status"] = ("incomplete_snapshot" if partial or changed or result["open_blocks"]
                            or exported.reset_seen else "snapshot")
    except Exception as error:
        # Parser exception messages can contain full log lines and player names.
        result["status"] = "read_or_parse_error"
        result["error"] = {"type": type(error).__name__, "line": line_number}
    return result


def write_snapshot(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--log", type=Path, help="Read one Power.log")
    source.add_argument("--logs", type=Path, help="Select the newest Power.log in a Logs directory")
    parser.add_argument("--client-app", type=Path, help="Read the installed macOS app build from Info.plist")
    parser.add_argument("--player-id", type=int, help="Override the unvalidated friendly-player inference")
    parser.add_argument("--out", type=Path, help="Atomically replace a local JSON snapshot")
    parser.add_argument("--watch", action="store_true", help="Refresh when the log changes or rotates")
    parser.add_argument("--interval", type=float, default=1.)
    args = parser.parse_args()
    if args.interval < .2:
        parser.error("--interval must be at least 0.2 seconds")
    build = None
    if args.client_app:
        with (args.client_app / "Contents/Info.plist").open("rb") as stream:
            build = plistlib.load(stream).get("CFBundleVersion")
    previous = object()
    try:
        while True:
            path = args.log if args.log is not None else newest_log(args.logs)
            try:
                stat = path.stat() if path is not None else None
                signature = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns) if stat else None
            except FileNotFoundError:
                signature = (str(path), "missing")
            if signature != previous:
                result = snapshot(path, player_id=args.player_id, installed_build=build)
                if args.out:
                    write_snapshot(args.out, result)
                    print(json.dumps({"status": result["status"], "out": str(args.out)}), flush=True)
                else:
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                previous = signature
            if not args.watch:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
