"""Reconstruct every actual decision boundary from a completed client log."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from hslog import packets
from .client_common import load_catalog, write_private_json
from .client_state import DiagnosticParser, VisibleExporter, last_game_lines, DUPLICATE
from .client_features import ClientFeatureAdapter
from .client_continuation_audit import audit as audit_continuations


def replay(path, player_id, races, output, runtime=None):
    lines, partial, changed, stat = last_game_lines(Path(path))
    parser = DiagnosticParser()
    for _, raw in lines:
        if raw.strip() and not DUPLICATE.match(raw): parser.read_line(raw.decode())
    parser.flush()
    if partial or changed or not parser.games: raise ValueError("Complete unchanged log required")
    _, catalog = load_catalog()
    adapter = runtime.adapter if runtime else ClientFeatureAdapter()
    rows, snapshots, menus = [], [], []
    class Replay(VisibleExporter):
        def capture(self, destination, event_time=None):
            started = time.perf_counter()
            snap = {"snapshot_id": f"{len(menus)}:{len(rows)}", "status": "snapshot", "game_status": "running",
                "entities": deepcopy(self.render_entities(catalog)), "perspective": {"player_id": player_id},
                "source": {"path": str(path), "game_start_line": lines[0][0]},
                "last_options": deepcopy(self.options), "last_choices": deepcopy(self.choices),
                "options_after_last_state_change": self.options_after_last_state_change,
                "choices_after_last_state_change": self.choices_after_last_state_change,
                "choice_packet_complete": self.choice_packet_complete, "decision_changes": list(self.decision_changes),
                "options_packet_complete": self.options_packet_complete,
                "public_history": deepcopy(self.public_history), "lobby_races": races,
                "lobby_races_source": "explicit_recording_metadata"}
            packet = adapter.convert(snap)
            row = {"index": len(destination), "turn": packet.actions.player.get("tags", {}).get("TURN", 0),
                "at": str(event_time) if event_time else None, "options_id": (self.options or {}).get("id"),
                **packet.report(), "adapter_ms": (time.perf_counter()-started)*1000}
            row["continuations"] = audit_continuations(packet, adapter.vocabulary, runtime)
            if runtime and packet.complete:
                try:
                    runtime.observe_client_history(snap)
                    row["decision"] = runtime.infer_client(packet)
                except Exception as error: row["error"] = type(error).__name__+": "+str(error)
            destination.append(row)
            return snap
        def handle_options(self, packet):
            super().handle_options(packet)
            self.capture(menus, packet.ts)
        def handle_send_option(self, packet):
            snapshots.append(self.capture(rows, packet.ts))
            super().handle_send_option(packet)
            rows[-1]["submitted"] = deepcopy(self.actions[-1])
        def handle_send_choices(self, packet):
            if type(packet) is packets.SendChoices: snapshots.append(self.capture(rows, packet.ts))
            super().handle_send_choices(packet)
            if type(packet) is packets.SendChoices: rows[-1]["submitted"] = deepcopy(self.actions[-1])
    Replay(parser.games[-1]).export()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output/"decision-frames.json.gz", "wt") as file: json.dump(snapshots, file, ensure_ascii=False)
    first_menus = {}
    for row in menus:
        if row["phase"] == "recruit" and row["turn"] > 0:
            first_menus.setdefault(str(row["turn"]), row)
    report = {"source": str(path), "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "scope": "client_recording_with_explicit_lobby_races", "player_id": player_id, "lobby_races": races,
        "decisions": len(rows), "complete": sum(r["status"] == "complete" for r in rows),
        "inferred": sum("decision" in r for r in rows), "errors": dict(Counter(r["error"] for r in rows if "error" in r)),
        "gaps": dict(Counter(g["code"] for r in rows for g in r["gaps"])),
        "adapter_p95_ms": float(np.percentile([r["adapter_ms"] for r in rows], 95)),
        "inference_p95_ms": float(np.percentile([r["decision"]["inference_ms"] for r in rows if "decision" in r], 95)) if runtime else None,
        "menu_frames": len(menus), "complete_menus": sum(r["status"] == "complete" for r in menus),
        "inferred_menus": sum("decision" in r for r in menus),
        "menu_errors": dict(Counter(r["error"] for r in menus if "error" in r)),
        "menu_gaps": dict(Counter(g["code"] for r in menus for g in r["gaps"])),
        "continuation_audit": {"legal_actions": sum(r["continuations"]["legal_actions"] for r in rows+menus),
            "stages": sum(r["continuations"]["stages"] for r in rows+menus),
            "battlecry_stages": sum(r["continuations"]["battlecry_stages"] for r in rows+menus),
            "random_projections": sum(r["continuations"]["random_projections"] for r in rows+menus),
            "errors": [e for r in rows+menus for e in r["continuations"]["errors"]]},
        "round_starts": first_menus, "rows": rows}
    write_private_json(output/"report.json", report)
    return {k: v for k, v in report.items() if k not in ("rows", "round_starts")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--player-id", type=int, required=True)
    p.add_argument("--races", nargs=5, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--infer", action="store_true")
    a = p.parse_args()
    runtime = None
    if a.infer:
        from .client_model import PolicyRuntime
        cfg = json.loads(Path("runs/client-model/selected.json").read_text())
        runtime = PolicyRuntime(cfg["checkpoint"], expected_sha256=cfg["sha256"],
            position_bundle=cfg["position_bundle"], position_sha256=cfg["position_sha256"])
    print(json.dumps(replay(a.path, a.player_id, a.races, a.out, runtime), ensure_ascii=False, indent=2))
