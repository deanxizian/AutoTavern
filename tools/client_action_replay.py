"""Audit action identities at every input packet in a completed recording."""

import argparse
from collections import Counter
import json
from pathlib import Path

from hslog import packets

from .client_actions import bind_actions, match_submission
from .client_common import load_catalog, write_private_json
from .client_state import DiagnosticParser, VisibleExporter, last_game_lines, DUPLICATE


def audit(path, player_id):
    rows, partial, changed, stat = last_game_lines(Path(path))
    parser = DiagnosticParser()
    for _, raw in rows:
        if raw.strip() and not DUPLICATE.match(raw):
            parser.read_line(raw.decode())
    parser.flush()
    if not parser.games or partial or changed:
        raise ValueError("Use a complete, unchanged recording")
    _, catalog = load_catalog()

    class Audit(VisibleExporter):
        def __init__(self, tree):
            super().__init__(tree)
            self.results = []

        def before(self):
            return bind_actions({"snapshot_id": str(len(self.results)), "status": "snapshot", "game_status": "running",
                "entities": self.render_entities(catalog), "perspective": {"player_id": player_id},
                "last_options": self.options, "last_choices": self.choices,
                "options_after_last_state_change": self.options_after_last_state_change,
                "choices_after_last_state_change": self.choices_after_last_state_change,
                "choice_packet_complete": self.choice_packet_complete})

        def handle_send_option(self, packet):
            mapping = self.before()
            super().handle_send_option(packet)
            self.results.append({**match_submission(mapping, self.actions[-1]), "event": self.actions[-1]})

        def handle_send_choices(self, packet):
            mapping = self.before() if type(packet) is packets.SendChoices else None
            super().handle_send_choices(packet)
            if mapping is not None:
                self.results.append({**match_submission(mapping, self.actions[-1]), "event": self.actions[-1]})

    exported = Audit(parser.games[-1]).export()
    return {"schema": "autotavern.action-binding-audit.v1", "source": str(Path(path).resolve()),
            "bytes": stat.st_size, "player_id": player_id, "submissions": len(exported.results),
            "counts": dict(Counter(r["status"] for r in exported.results)), "rows": exported.results,
            "execution_verified": False, "live_ready": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--player-id", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.path, args.player_id)
    write_private_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
