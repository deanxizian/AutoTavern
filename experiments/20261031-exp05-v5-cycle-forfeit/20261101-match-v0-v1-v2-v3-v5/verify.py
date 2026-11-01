"""Recount scores, fair exposure and immutable checkpoints from sealed lobby records."""

from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / "pyproject.toml").is_file())
GROUPS = ("v0", "v1", "v2", "v3", "v5")


def main():
    result = json.loads((FOLDER / "results.json").read_text())
    config = json.loads((FOLDER / "config.json").read_text())
    assert result["config"] == config
    rows = [json.loads(line) for line in (FOLDER / "games.jsonl").read_text().splitlines()]
    assert result["status"] == "completed" and len(rows) == config["games"] == 320
    assert len({(r["lineup_id"], r["lane"], r["episode_index"]) for r in rows}) == 320
    assert Counter(r["lineup_id"] for r in rows) == dict.fromkeys(range(10), 32)
    for row in rows:
        assert row["episode_index"] == 0 and 0 <= row["lane"] < 32
        assert row["lineup"] == config["lineups"][row["lineup_id"]]
        assert sorted(sum(row["placements"].values(), [])) == list(range(1, 9))
        assert sorted(sum(row["model_seats"].values(), []) + row["v0_seats"]) == list(range(8))
        assert all(len(row["placements"][name]) == row["lineup"][name] for name in GROUPS)
        assert all(len(row["model_seats"][name]) == len(row["model_actions"][name])
                   == row["lineup"][name] for name in GROUPS[1:])
    for name, summary in result["groups"].items():
        places = [rank for row in rows for rank in row["placements"][name]]
        assert len(places) == summary["player_games"] == 512
        assert sum(places) / len(places) == summary["mean_placement"]
        assert sum(p == 1 for p in places) / len(places) == summary["first_rate"]
        assert sum(p <= 4 for p in places) / len(places) == summary["top4_rate"]
        assert dict(sorted(Counter(places).items())) == {
            int(k): v for k, v in summary["placement_counts"].items()}
    for a, b in itertools.combinations(GROUPS, 2):
        ahead = sum(x < y for row in rows for x in row["placements"][a] for y in row["placements"][b])
        total = sum(len(row["placements"][a]) * len(row["placements"][b]) for row in rows)
        summary = result["pairwise"][f"{a}_vs_{b}"]
        assert total == summary["pair_comparisons"] == 800
        assert ahead / total == summary["ahead_rate_first_vs_second"]
    assert result["rejected_actions"] == result["optimizer_updates"] == 0
    assert result["training_started"] is False and result["weights_unchanged"] is True
    for spec in config["models"].values():
        assert hashlib.sha256((ROOT / spec["path"]).read_bytes()).hexdigest() == spec["sha256"]
    report = {"status": "passed", "completed_lobbies": 320, "player_games_per_group": 512,
              "lineup_quotas_and_exact_ranks": True, "all_metrics_independently_recounted": True,
              "source_weights_unchanged": True, "no_training_or_rejected_actions": True,
              "first_role_counts": dict(Counter(r["first_role"] for r in rows))}
    (FOLDER / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
