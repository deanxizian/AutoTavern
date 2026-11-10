"""Recount completed lobby records independently of the reporting aggregator."""

from collections import Counter
import hashlib
import json
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / "pyproject.toml").exists())


def main():
    result = json.loads((FOLDER / "results.json").read_text())
    config = result["config"]
    rows = [json.loads(line) for line in (FOLDER / "games.jsonl").read_text().splitlines()]
    assert result["status"] == "completed" and len(rows) == config["games"] == 256
    assert len({(r["lane"], r["episode_index"]) for r in rows}) == len(rows)
    assert set(Counter(r["lane"] for r in rows).values()) == {4}
    for row in rows:
        assert all(len(values) == 2 for values in row["placements"].values())
        assert sorted(sum(row["placements"].values(), [])) == list(range(1, 9))
        seats = sum(row["model_seats"].values(), []) + row["scripted_seats"]
        assert sorted(seats) == list(range(8))
    for name, summary in result["groups"].items():
        places = [rank for row in rows for rank in row["placements"][name]]
        assert len(places) == summary["player_games"] == 512
        assert sum(places) / len(places) == summary["mean_placement"]
        assert sum(p == 1 for p in places) / len(places) == summary["first_rate"]
        assert sum(p <= 4 for p in places) / len(places) == summary["top4_rate"]
    for newer, older in (("v2", "v1"), ("v3", "v1"), ("v3", "v2")):
        ahead = sum(a < b for row in rows for a in row["placements"][newer]
                    for b in row["placements"][older])
        assert ahead / (4 * len(rows)) == result[f"{newer}_vs_{older}"]["pairwise_ahead_rate"]
    assert result["rejected_actions"] == result["optimizer_updates"] == 0
    assert result["training_started"] is False and result["weights_unchanged"] is True
    for spec in config["models"].values():
        assert hashlib.sha256((ROOT / spec["path"]).read_bytes()).hexdigest() == spec["sha256"]
    report = {"status": "passed", "completed_lobbies": len(rows), "player_games_per_group": 512,
              "unique_lobbies_and_lane_quotas": True, "exact_ranks_and_seats": True,
              "group_and_pairwise_metrics_recounted": True, "source_weights_unchanged": True,
              "no_training_or_rejected_actions": True,
              "first_policy_priority_counts": dict(Counter(
                  (r["lane"] + r["episode_index"]) % 3 for r in rows))}
    (FOLDER / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
