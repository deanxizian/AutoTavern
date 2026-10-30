"""Replay timestamped log writes without a client or real-time waiting.

The captured clock drives the 25 ms quiet-EOF rule. CPU timings measure actual
incremental reads; they do not claim screen-to-advice latency.
"""
import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import re
import tempfile

import numpy as np
from .client_actions import bind_actions
from .client_common import write_private_json
from .client_features import ClientFeatureAdapter
from .client_state import snapshot
from .client_stream import IncrementalReader


def replay(path, player_id, races, output):
    raw = Path(path).read_bytes()
    pattern = re.compile(rb"^[A-Z] (\d{2}):(\d{2}):(\d{2}\.\d+)")
    def stamp(line):
        match = pattern.match(line)
        if not match: raise ValueError("Recording contains a line without a timestamp")
        h, m, s = match.groups()
        return int(h)*3600 + int(m)*60 + float(s)
    adapter, clock = ClientFeatureAdapter(), [0.]
    menus, starts, timings, read_bytes = {}, {}, [], 0
    def inspect(snap):
        if not snap.get("entities") or snap.get("game_status") == "complete": return
        a = bind_actions(snap)
        if a.phase not in ("recruit", "hero_select"): return
        choice = snap.get("last_choices") or {}
        if choice and not choice.get("selection_sent_or_resolved"):
            if not snap.get("choice_packet_complete"): return
            key = "choice:" + str(choice["id"])
        else:
            if not snap.get("options_after_last_state_change") or not snap.get("options_packet_complete"): return
            key = "options:" + str(snap["last_options"]["id"])
        if menus.get(key, {}).get("complete"): return
        packet = adapter.convert({**snap, "lobby_races": races})
        row = {"at_log_seconds": clock[0], "turn": a.player.get("tags", {}).get("TURN", 0),
            "complete": bool(packet.complete), "gaps": packet.gaps}
        menus[key] = row
        if packet.complete and a.phase == "recruit" and row["turn"] > 0:
            starts.setdefault(str(row["turn"]), {"menu": key, **row})
    with tempfile.TemporaryDirectory(prefix="client-stream-replay-") as directory:
        target = Path(directory) / "Power.log"
        target.touch(mode=0o600)
        reader = IncrementalReader(player_id=player_id, clock=lambda: clock[0])
        previous = None
        for at, lines in groupby(raw.splitlines(keepends=True), stamp):
            if previous is not None and at - previous >= .025:
                clock[0] = previous + .025001
                inspect(reader.read(target))
            clock[0] = max(clock[0], at)
            with target.open("ab") as stream: stream.write(b"".join(lines))
            actual = reader.read(target)
            if actual.get("status") == "read_or_parse_error": raise ValueError(actual["error"])
            timings.append(actual["performance"]["total_ms"])
            read_bytes += actual["performance"]["read_bytes"]
            inspect(actual)
            previous = at
        clock[0] += .03
        actual = reader.read(target)
        inspect(actual)
        expected = snapshot(target, player_id=player_id)
        for key in ("entities", "players", "last_options", "last_choices", "action_counts", "public_history", "game_status"):
            if actual[key] != expected[key]: raise AssertionError("Stream/batch mismatch: " + key)
    report = {"source": str(path), "scope": "timestamped_offline_stream_replay", "source_bytes": len(raw),
        "total_stream_bytes_read": read_bytes, "chunks": len(timings), "final_batch_equivalence": True,
        "menu_count": len(menus), "complete_menus": sum(m["complete"] for m in menus.values()),
        "remaining_gaps": dict(Counter(g["code"] for m in menus.values() if not m["complete"] for g in m["gaps"])),
        "round_starts": starts, "read_median_ms": float(np.median(timings)),
        "read_p95_ms": float(np.percentile(timings, 95)), "read_max_ms": max(timings), "menus": menus}
    write_private_json(output, report)
    return {k: v for k, v in report.items() if k not in ("menus", "round_starts")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--player-id", type=int, required=True)
    p.add_argument("--races", nargs=5, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(replay(a.path, a.player_id, a.races, a.out), ensure_ascii=False, indent=2))
