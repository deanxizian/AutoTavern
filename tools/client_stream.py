"""Append-only Power.log reader. Parser state and packet cursors survive each poll.

No snapshot cache is persisted: a new session, rotation or detected rewrite rebuilds
from the file. Partial lines and unfinished packets never become executable advice.
The hslog internals used here are pinned to 1.20.0 and checked against batch replay.
"""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import time

from hslog import packets
from hslog.export import FriendlyPlayerExporter
from hearthstone.enums import BlockType

if __package__:
    from .client_common import DEFAULT_MAX_AGE, load_catalog
    from .client_state import (CREATE, DUPLICATE, MAX_GAME_BYTES, DiagnosticParser, VisibleExporter,
        base_snapshot, file_signature, populate_snapshot, validate_snapshot)
else:
    from client_common import DEFAULT_MAX_AGE, load_catalog
    from client_state import (CREATE, DUPLICATE, MAX_GAME_BYTES, DiagnosticParser, VisibleExporter,
        base_snapshot, file_signature, populate_snapshot, validate_snapshot)


MUTABLE = (packets.CreateGame, packets.FullEntity, packets.ShowEntity, packets.ChangeEntity,
           packets.Options, packets.Choices, packets.SendChoices, packets.ChosenEntities)


def packet_revision(packet):
    if isinstance(packet, packets.CreateGame):
        return len(packet.tags), tuple(len(p.tags) for p in packet.players)
    if isinstance(packet, packets.Options):
        def shape(options):
            return tuple(shape(p.options) for p in options)
        return shape(packet.options)
    if isinstance(packet, (packets.Choices, packets.SendChoices, packets.ChosenEntities)):
        return len(packet.choices), getattr(packet, "source", None), packet.entity
    return len(packet.tags), packet.card_id


class PacketCursor:
    """Walk only new tree nodes; revisit a mutable tail when its tags/options grow."""
    def __init__(self, tree):
        self.stack = [[tree, 0]]
        self.exported = VisibleExporter(tree)
        self.friendly = FriendlyPlayerExporter(tree)
        self.tail = None

    def drain(self):
        while self.stack:
            node, index = self.stack[-1]
            if index == len(node.packets):
                if len(self.stack) == 1 or not node.ended:
                    break
                self.stack.pop()
                continue
            packet = node.packets[index]
            if isinstance(packet, (packets.Block, packets.SubSpell)):
                self.stack[-1][1] += 1
                if isinstance(packet, packets.Block):
                    self.exported.begin_block(packet)
                self.stack.append([packet, 0])
                continue
            mutable = isinstance(packet, MUTABLE)
            stamp = packet_revision(packet) if mutable else None
            if self.tail is None or self.tail[0] is not packet or self.tail[1] != stamp:
                self.exported.export_packet(packet)
                if self.friendly.friendly_player is None:
                    if isinstance(packet, packets.CreateGame):
                        self.friendly = FriendlyPlayerExporter(self.stack[0][0])
                    self.friendly.export_packet(packet)
            if mutable and index == len(node.packets) - 1 and not getattr(node, "ended", False):
                if isinstance(packet, packets.Choices): self.exported.choice_packet_complete = False
                if isinstance(packet, packets.Options): self.exported.options_packet_complete = False
                self.tail = packet, stamp
                break
            if isinstance(packet, packets.Choices): self.exported.choice_packet_complete = True
            if isinstance(packet, packets.Options): self.exported.options_packet_complete = True
            self.tail = None
            self.stack[-1][1] += 1

    @property
    def open_blocks(self):
        return sum(not frame[0].ended for frame in self.stack[1:])


class IncrementalReader:
    def __init__(self, *, player_id=None, installed_build=None, max_age=DEFAULT_MAX_AGE, clock=time.monotonic):
        self.player_id, self.installed_build, self.max_age = player_id, installed_build, max_age
        self.clock = clock
        self.meta, self.cards = load_catalog()
        self.path = None
        self._reset("initial")

    def _reset(self, reason):
        self.offset = self.line_number = self.game_start = self.game_bytes = 0
        self.last_append_at = 0.
        self.pending = self.prefix = self.anchor = b""
        self.identity = self.signature = self.cached = self.error = None
        self.parser = self.cursor = None
        self.digest = hashlib.sha256()
        self.reset_reason = reason

    def _feed(self, data):
        data = self.pending + data
        end = data.rfind(b"\n") + 1
        self.pending = data[end:]
        rows = data[:end].splitlines(keepends=True)
        latest_start = next((i for i in range(len(rows) - 1, -1, -1)
                             if b"CREATE_GAME" in rows[i] and CREATE.search(rows[i])), 0)
        self.line_number += latest_start
        for raw in rows[latest_start:]:
            self.line_number += 1
            if b"CREATE_GAME" in raw and CREATE.search(raw):
                self.parser = DiagnosticParser()
                self.cursor = None
                self.error = None
                self.game_start, self.game_bytes = self.line_number, 0
                self.digest = hashlib.sha256()
            if not self.game_start:
                continue
            self.digest.update(raw)
            self.game_bytes += len(raw)
            if self.game_bytes > MAX_GAME_BYTES:
                self.error = {"type": "ValueError", "line": self.line_number}
            if self.error:
                continue
            try:
                # These duplicate execution logs are not consumed by hslog's GameState handlers.
                if raw.strip() and not DUPLICATE.match(raw):
                    self.parser.read_line(raw.decode("utf-8"))
            except Exception as error:
                self.error = {"type": type(error).__name__, "line": self.line_number}

    def read(self, path, *, now=None):
        started = time.perf_counter()
        path = Path(path).resolve() if path else None
        if path != self.path:
            self._reset("session_changed" if self.path else "initial")
            self.path = path
        result = base_snapshot(self.meta, self.installed_build, now)
        read_bytes = 0
        parse_ms = 0.
        cache_hit = False
        try:
            if path is None:
                return validate_snapshot(result, now=now, max_age=self.max_age)
            result["source"] = {"path": str(path)}
            stat = path.stat()
            signature = file_signature(stat)
            if self.cached and signature == self.signature:
                result = dict(self.cached)
                result["source"] = dict(result["source"])
                result["captured_at"] = (now or datetime.now(timezone.utc)).isoformat()
                cache_hit = True
                # These packets have no end marker. A following packet proves
                # completion immediately; a quiet, whole-line EOF is accepted
                # after 25 ms. Never infer from a growing partial menu.
                if self.cursor and self.cursor.tail and self.clock() - self.last_append_at >= .025 and not self.pending:
                    tail = self.cursor.tail[0]
                    if isinstance(tail, packets.Choices):
                        result["choice_packet_complete"] = self.cursor.exported.choice_packet_complete = True
                    if isinstance(tail, packets.Options):
                        result["options_packet_complete"] = self.cursor.exported.options_packet_complete = True
            else:
                with path.open("rb") as stream:
                    before = os.fstat(stream.fileno())
                    identity = before.st_dev, before.st_ino
                    if self.identity is not None:
                        reason = None
                        if identity != self.identity: reason = "replaced"
                        elif before.st_size < self.offset: reason = "truncated"
                        elif before.st_size == self.offset and self.signature != file_signature(before): reason = "rewritten"
                        elif self.offset:
                            prefix = stream.read(len(self.prefix))
                            stream.seek(self.offset - len(self.anchor))
                            if prefix != self.prefix or stream.read(len(self.anchor)) != self.anchor:
                                reason = "rewritten"
                        if reason: self._reset(reason)
                    stream.seek(self.offset)
                    data = stream.read(before.st_size - self.offset)
                    read_bytes = len(data)
                    self.offset += read_bytes
                    stream.seek(0)
                    self.prefix = stream.read(min(self.offset, 128))
                    stream.seek(max(0, self.offset - 256))
                    self.anchor = stream.read(min(self.offset, 256))
                    after = os.fstat(stream.fileno())
                current = path.stat()
                changed = file_signature(before) != file_signature(after) or file_signature(before) != file_signature(current)
                self.identity, self.signature = identity, file_signature(before)
                tick = time.perf_counter()
                self._feed(data)
                if data: self.last_append_at = self.clock()
                if self.parser and not self.error:
                    if self.cursor is None: self.cursor = PacketCursor(self.parser.games[-1])
                    self.cursor.drain()
                parse_ms = (time.perf_counter() - tick) * 1000
                source = result["source"]
                source.update(bytes=before.st_size, mtime_ns=before.st_mtime_ns,
                    device=before.st_dev, inode=before.st_ino,
                    modified_at=datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
                    changed_while_reading=changed, partial_trailing_line=bool(self.pending))
                if not self.game_start:
                    result["status"] = "empty_log" if before.st_size == 0 else "no_complete_game_start"
                elif self.error:
                    result.update(status="read_or_parse_error", error=dict(self.error))
                else:
                    source.update(game_start_line=self.game_start, last_complete_line=self.line_number,
                                  parsed_game_sha256=self.digest.hexdigest())
                    player = self.player_id if self.player_id is not None else self.cursor.friendly.friendly_player
                    perspective = "explicit_player_id" if self.player_id is not None else "hslog_inference_unvalidated_for_this_game"
                    populate_snapshot(result, self.cursor.exported, self.parser, self.cards,
                                      player, perspective, self.cursor.open_blocks)
                if changed:
                    # Never reuse an uncertain prefix after concurrent writes or replacement.
                    self._reset("changed_during_read")
        except FileNotFoundError:
            self._reset("missing")
            result["status"] = "missing_log"
        except Exception as error:
            result.update(status="read_or_parse_error", error={"type": type(error).__name__, "line": self.line_number})
        result = validate_snapshot(result, now=now, max_age=self.max_age)
        result["performance"] = {"read_bytes": read_bytes, "parse_ms": round(parse_ms, 3),
            "total_ms": round((time.perf_counter() - started) * 1000, 3), "cache_hit": cache_hit,
            "reset_reason": self.reset_reason}
        self.cached = result if self.signature is not None else None
        return result
