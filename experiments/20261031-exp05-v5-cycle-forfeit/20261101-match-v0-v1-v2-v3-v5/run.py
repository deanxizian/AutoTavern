"""Frozen V0/V1/V2/V3/V5 match, with balanced eight-player lineup exposures."""

import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

import taverngym as tg
from taverngym.learning.experiment import load_frozen_policy, now, write_json
from taverngym.learning.tracking import WandbTracker


FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / "pyproject.toml").is_file())
GROUPS = ("v0", "v1", "v2", "v3", "v5")
MODEL_GROUPS = GROUPS[1:]
METRICS = ("first_rate", "top4_rate", "mean_placement", "lobby_win_share")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, confidence=False):
    """Use player-entry ratios and resample whole lobbies within fixed lineup strata."""
    result = {"lobbies": len(rows), "groups": {}, "pairwise": {}}
    indices = None
    if confidence:
        rng = np.random.default_rng(61101502)
        indices = np.concatenate([
            rng.choice([i for i, row in enumerate(rows) if row["lineup_id"] == lineup],
                       size=(10000, sum(row["lineup_id"] == lineup for row in rows)))
            for lineup in sorted({row["lineup_id"] for row in rows})
        ], axis=1)

    def ratio(numerators, denominators):
        numerators = np.asarray(numerators)
        denominators = np.asarray(denominators)
        value = float(numerators.sum() / denominators.sum())
        ci = None if indices is None else np.quantile(
            numerators[indices].sum(axis=1) / denominators[indices].sum(axis=1),
            [0.025, 0.975]).tolist()
        return value, ci

    for name in GROUPS:
        values = [row["placements"][name] for row in rows]
        counts = [len(v) for v in values]
        flat = sum(values, [])
        metrics = {"player_games": len(flat), "first_places": flat.count(1),
                   "placement_counts": {str(i): flat.count(i) for i in range(1, 9)}}
        for key, numerators, denominators in (
            ("first_rate", [v.count(1) for v in values], counts),
            ("top4_rate", [sum(p <= 4 for p in v) for v in values], counts),
            ("mean_placement", [sum(v) for v in values], counts),
            ("lobby_win_share", [int(1 in v) for v in values], np.ones(len(rows))),
        ):
            value, ci = ratio(numerators, denominators)
            metrics[key] = value
            if ci is not None:
                metrics[f"{key}_ci95"] = ci
        result["groups"][name] = metrics
    for a, b in itertools.combinations(GROUPS, 2):
        wins = [sum(x < y for x in row["placements"][a] for y in row["placements"][b])
                for row in rows]
        totals = [len(row["placements"][a]) * len(row["placements"][b]) for row in rows]
        value, ci = ratio(wins, totals)
        metrics = {"ahead_rate_first_vs_second": value, "pair_comparisons": sum(totals)}
        if ci is not None:
            metrics["ahead_rate_ci95"] = ci
        result["pairwise"][f"{a}_vs_{b}"] = metrics
    if confidence:
        result["confidence_method"] = (
            "10,000 stratified whole-lobby bootstrap samples, preserving 32 lobbies per lineup; "
            "seat-weighted ratios; percentile 95% intervals")
    assert sum(g["first_places"] for g in result["groups"].values()) == len(rows)
    return result


def scalars(rows):
    summary = summarize(rows)
    return {"evaluated_lobbies": len(rows),
            **{f"test/{name}_{key}": group[key]
               for name, group in summary["groups"].items() for key in METRICS}}


def main():
    config = json.loads((FOLDER / "config.json").read_text())
    expected = [{n: 1 + int(n in extra) for n in GROUPS}
                for extra in itertools.combinations(GROUPS, 3)]
    assert config["lineups"] == expected
    assert config["games"] == 320 and config["parallel_games"] == config["games_per_lineup"] == 32
    if (FOLDER / "games.jsonl").exists():
        raise FileExistsError("Refusing to overwrite an existing match")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(config["seed"])
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert tg.build_is_current() is True
    policies = {}
    for name, spec in config["models"].items():
        path = ROOT / spec["path"]
        assert digest(path) == spec["sha256"]
        policies[name] = load_frozen_policy(path, "cuda", forward_precision="bf16")
        assert {str(p.dtype) for p in policies[name].parameters()} == {"torch.float32"}
        assert not any(p.requires_grad for p in policies[name].parameters())

    started = last_report = time.monotonic()
    metadata = {"status": "running", "started_at": now(), "config": "config.json",
                "training_started": False, "optimizer_updates": 0, "weights_uploaded": False,
                "native_build": tg.build_info(), "models": config["models"]}
    write_json(FOLDER / "experiment.json", metadata)
    tracker = None
    exit_code = 1
    rows = []
    deferred = total_vector_steps = 0
    try:
        tracker = WandbTracker(FOLDER, config, kind="frozen-policy-balanced-lobby-match")
        tracker.run.define_metric("evaluated_lobbies")
        tracker.run.define_metric("test/*", step_metric="evaluated_lobbies")
        tracking = json.loads((FOLDER / "wandb.json").read_text())
        tracking["x_axis"] = "evaluated_lobbies"
        write_json(FOLDER / "wandb.json", tracking)
        with (FOLDER / "games.jsonl").open("x", buffering=1) as stream:
            for lineup_id, lineup in enumerate(config["lineups"]):
                n_games = config["parallel_games"]
                role_ids = [j for repeat in range(2) for j, name in enumerate(MODEL_GROUPS)
                            if lineup[name] > repeat]
                roles = len(role_ids)
                identity = np.array([np.roll(role_ids, -i) for i in range(n_games)])
                env = tg.BatchEnv(n_games, seed=config["seed"] + lineup_id * 100003,
                                  controlled=range(roles), threads=config["threads"],
                                  config=config["simulator_config"], rotate_seats=True)
                lane = np.arange(env.n_slots) // roles
                places = np.zeros((n_games, roles), dtype=np.int64)
                actions_taken = np.zeros_like(places)
                seats = env.slot_seats.reshape(n_games, roles).copy()
                active = np.ones(n_games, dtype=bool)
                for vector_step in range(20000):
                    if not active.any():
                        break
                    obs = env.observe()
                    ready = obs["ready"] & active[lane]
                    actions = np.zeros(env.n_slots, dtype=np.int64)
                    for policy_id, name in enumerate(MODEL_GROUPS):
                        selected = np.flatnonzero(ready & (identity.reshape(-1) == policy_id))
                        if not len(selected):
                            continue
                        inputs = {k: v[selected] for k, v in obs.items() if k != "ready"}
                        with torch.inference_mode():
                            predicted, _ = policies[name].predict(inputs, deterministic=False)
                        assert inputs["action_mask"][np.arange(len(selected)), predicted].all()
                        actions[selected] = predicted
                    _, dones, info = env.step(actions, active=active)
                    if info["rejected"].any() or ((info["placement"] == 0) & dones).any():
                        for i in np.unique(lane[info["rejected"] | dones]):
                            write_json(FOLDER / f"guard-{lineup_id}-{i}.json", json.loads(env.save(int(i))))
                        raise RuntimeError("Native rejection or terminal without a real placement")
                    deferred += int(info["deferred"].sum())
                    actions_taken += (ready & ~info["deferred"]).reshape(n_games, roles)
                    terminal = dones.reshape(n_games, roles)
                    assert not (terminal & (places > 0)).any(), "Duplicate terminal report"
                    places[terminal] = info["placement"].reshape(n_games, roles)[terminal]
                    for i in np.flatnonzero(active & (places > 0).all(axis=1)):
                        assert len(set(places[i].tolist())) == roles
                        row = {"lineup_id": lineup_id, "lineup": lineup, "lane": int(i),
                               "episode_index": 0,
                               "placements": {name: places[i, identity[i] == j].tolist()
                                              for j, name in enumerate(MODEL_GROUPS)},
                               "model_seats": {name: seats[i, identity[i] == j].tolist()
                                               for j, name in enumerate(MODEL_GROUPS)},
                               "model_actions": {name: actions_taken[i, identity[i] == j].tolist()
                                                 for j, name in enumerate(MODEL_GROUPS)},
                               "first_role": MODEL_GROUPS[int(identity[i, 0])],
                               "v0_seats": sorted(set(range(8)) - set(seats[i].tolist()))}
                        row["placements"]["v0"] = sorted(set(range(1, 9)) - set(places[i].tolist()))
                        assert all(len(row["placements"][name]) == lineup[name] for name in GROUPS)
                        assert sorted(sum(row["placements"].values(), [])) == list(range(1, 9))
                        assert env.episode_index(int(i)) == 1
                        rows.append(row)
                        stream.write(json.dumps(row) + "\n")
                        active[i] = False
                    total_vector_steps += 1
                    if time.monotonic() - last_report >= 15:
                        progress = {"status": "running", "updated_at": now(),
                                    "completed_lobbies": len(rows), "target_lobbies": config["games"],
                                    "lineup_id": lineup_id, "vector_steps": total_vector_steps,
                                    "elapsed_seconds": time.monotonic() - started}
                        write_json(FOLDER / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
                        last_report = time.monotonic()
                else:
                    raise RuntimeError("Match exceeded its bounded vector-step budget")
                assert not active.any()
                tracker.log(scalars(rows), 0)
                del env
        assert len(rows) == config["games"]
        assert all(digest(ROOT / spec["path"]) == spec["sha256"] for spec in config["models"].values())
        result = {"status": "completed", "completed_at": now(), "config": config,
                  **summarize(rows, confidence=True), "elapsed_seconds": time.monotonic() - started,
                  "vector_steps": total_vector_steps, "deferred_actions": deferred,
                  "rejected_actions": 0, "weights_unchanged": True,
                  "training_started": False, "optimizer_updates": 0,
                  "limitations": [
                      "Results apply to these ten balanced eight-player lineups and stochastic BF16 inference.",
                      "Each version has 512 player entries across 320 lobbies; correlated seats are not independent.",
                      "V0 aggregate ranks are the exact unordered complement of neural-seat ranks.",
                      "Pairwise ahead rate compares final placement, not combat outcomes.",
                      "Independent test results do not select checkpoints or training opponents."]}
        write_json(FOLDER / "results.json", result)
        tracker.log(scalars(rows), 0)
        metadata.update(status="completed", completed_at=now(), results="results.json",
                        completed_lobbies=len(rows), elapsed_seconds=result["elapsed_seconds"])
        write_json(FOLDER / "experiment.json", metadata)
        write_json(FOLDER / "progress.json", metadata)
        print(json.dumps(result["groups"]), flush=True)
        exit_code = 0
    except BaseException as exc:
        metadata.update(status="failed", failed_at=now(), error=f"{type(exc).__name__}: {exc}",
                        completed_lobbies=len(rows))
        write_json(FOLDER / "experiment.json", metadata)
        raise
    finally:
        if tracker is not None:
            tracker.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
