"""Replay a private recording, verify every captured boundary, and measure latency."""

import argparse
import gzip
import json
from pathlib import Path
import statistics
import tempfile
import time

if __package__:
    from .client_common import write_private_json
    from .client_state import snapshot
    from .client_stream import IncrementalReader
else:
    from client_common import write_private_json
    from client_state import snapshot
    from client_stream import IncrementalReader


KEYS = ("status", "players", "perspective", "last_options", "last_choices",
        "options_after_last_state_change", "choices_after_last_state_change", "open_blocks", "reset_seen")


def distribution(samples):
    ordered = sorted(samples)
    return {"count": len(samples), "median_ms": round(statistics.median(samples), 3),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(len(ordered) * .95))], 3),
            "max_ms": round(max(samples), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.recording / "recording-end.json").read_text())
    source = next(Path(f["copy"]) for f in manifest["files"] if Path(f["source"]).name == "Power.log")
    raw = source.read_bytes()
    timeline = [json.loads(row) for row in (args.recording / "timeline.jsonl").read_text().splitlines()]
    cold = []
    for _ in range(3):
        reader = IncrementalReader()
        started = time.perf_counter(); reader.read(source); cold.append((time.perf_counter() - started) * 1000)
    idle = []
    for _ in range(100):
        started = time.perf_counter(); reader.read(source); idle.append((time.perf_counter() - started) * 1000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    timings, bytes_read = [], []
    with tempfile.TemporaryDirectory(prefix="replay-", dir=args.out.parent) as directory:
        path = Path(directory) / "Power.log"
        path.touch()
        reader = IncrementalReader()
        offset = 0
        for row in timeline:
            end = row["source_bytes"]
            with path.open("ab") as stream:
                stream.write(raw[offset:end])
            started = time.perf_counter(); actual = reader.read(path); elapsed = (time.perf_counter() - started) * 1000
            timings.append(elapsed)
            bytes_read.append(actual["performance"]["read_bytes"])
            expected = snapshot(path)
            with gzip.open(args.recording / f"snapshot-{row['sequence']:05}.json.gz", "rt") as stream:
                recorded = json.load(stream)
            for key in KEYS:
                if actual.get(key) != expected.get(key) or actual.get(key) != recorded.get(key):
                    raise AssertionError(f"Replay mismatch at sample {row['sequence']}, field {key}")
            legacy_entities = [{k: v for k, v in e.items() if k != "catalog_category"} for e in actual.get("entities", [])]
            if legacy_entities != recorded.get("entities", []):
                raise AssertionError(f"Recorded entities differ at sample {row['sequence']}")
            for key in ("entities", "snapshot_id", "recent_actions", "action_counts", "parser_diagnostics"):
                if actual.get(key) != expected.get(key):
                    raise AssertionError(f"Batch equivalence failed at sample {row['sequence']}, field {key}")
            offset = end
    result = {"source_bytes": len(raw), "captured_boundaries_verified": len(timeline),
        "exact_recorded_state_match": True, "batch_equivalence": True,
        "cold_full_read": distribution(cold), "idle_read": distribution(idle),
        "incremental_updates": distribution(timings[1:]), "total_stream_bytes_read": sum(bytes_read),
        "action_counts": actual["action_counts"], "parser_diagnostics": actual["parser_diagnostics"],
        "unknown_gameplay_cards": actual["unknown_card_ids"],
        "internal_entities": len(actual["unmapped_internal_ids"]), "skins": len(actual["unmapped_skin_ids"]),
        "game_status": actual["game_status"], "incremental_samples_ms": timings}
    write_private_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("incremental_samples_ms", "parser_diagnostics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
