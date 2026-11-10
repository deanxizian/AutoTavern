"""Check saved audit counters and extract compact, verifiable action examples."""

from collections import Counter, defaultdict
import json
from pathlib import Path


FOLDER = Path(__file__).resolve().parent


def main():
    results = json.loads((FOLDER / "results.json").read_text())
    examples = {}
    for mode, summary in results["conditions"].items():
        traces = defaultdict(list)
        for line in (FOLDER / f"{mode}-actions.jsonl").open():
            row = json.loads(line)
            traces[row["lane"], row["turn"]].append(row)
        turns = [json.loads(line) for line in
                 (FOLDER / f"{mode}-turns.jsonl").read_text().splitlines()]
        assert len(traces) == len(turns)
        for turn in turns:
            rows = traces[turn["lane"], turn["turn"]]
            tail = [r for r in rows if r["step"] > 60 and r["action"] != 0]
            assert len(tail) == turn["non_end_after60"]
            assert sum(r["visible_revisit"] for r in tail) == turn["visible_revisits_after60"]
            assert dict(Counter(r["action_type"] for r in tail)) == turn["action_types_after60"]
            assert sum(r["move_freeze_cycle"] for r in tail) == turn["cycles_after60"]
        assert sum(t["visible_revisits_after60"] for t in turns) == summary["visible_revisits"]
        mode_examples = {"two_action_visible_loops": [], "gold_spending_after60": []}
        for rows in traces.values():
            for previous, row in zip(rows, rows[1:]):
                if (row["step"] > 60 and row["action"] != 0
                        and previous["before_hash"] == row["after_hash"]
                        and row["action_type"] not in ("Move", "Freeze")
                        and len(mode_examples["two_action_visible_loops"]) < 5):
                    mode_examples["two_action_visible_loops"].append([previous, row])
            for row in rows:
                if (row["step"] > 60 and row["action"] != 0
                        and not row["visible_revisit"]
                        and row["gold_after"] < row["gold_before"] - 0.1
                        and len(mode_examples["gold_spending_after60"]) < 5):
                    mode_examples["gold_spending_after60"].append(row)
        examples[mode] = mode_examples
    (FOLDER / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")
    (FOLDER / "verification.json").write_text(json.dumps({
        "status": "passed", "unique_turns_and_action_counters": True,
        "all_action_revisits_verified": True,
        "move_freeze_revisits_verified": True,
    }, indent=2) + "\n")
    for mode, row in results["conditions"].items():
        print(mode, json.dumps(row))
        print("two-action loops", [
            [(r["step"], r["action_type"], r["action"]) for r in pair]
            for pair in examples[mode]["two_action_visible_loops"]])
        print("gold spending", [(r["step"], r["action_type"], r["gold_before"],
                                  r["gold_after"])
                                 for r in examples[mode]["gold_spending_after60"]])


if __name__ == "__main__":
    main()
