"""Frozen-policy eight-player matches: two native scripts, three V1, three V2."""

import hashlib
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
GROUPS = ("v1", "v2", "scripted")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows, confidence=False):
    """The independent sampling unit is a lobby, not its correlated player seats."""
    placements = {name: np.array([row["placements"][name] for row in rows]) for name in GROUPS}
    result = {"lobbies": len(rows), "groups": {}}
    indices = None
    if confidence:
        indices = np.random.default_rng(631017).integers(0, len(rows), (10000, len(rows)))

    def interval(lobby_values):
        return np.quantile(lobby_values[indices].mean(axis=1), [0.025, 0.975]).tolist()

    for name, values in placements.items():
        wins = (values == 1).any(axis=1)
        metrics = {
            "seats_per_lobby": values.shape[1], "player_games": int(values.size),
            "first_places": int((values == 1).sum()), "first_rate": float((values == 1).mean()),
            "top4_rate": float((values <= 4).mean()), "mean_placement": float(values.mean()),
            "lobby_win_share": float(wins.mean()),
            "placement_counts": {str(i): int((values == i).sum()) for i in range(1, 9)},
        }
        if confidence:
            metrics["first_rate_ci95"] = interval((values == 1).mean(axis=1))
            metrics["mean_placement_ci95"] = interval(values.mean(axis=1))
            metrics["lobby_win_share_ci95"] = interval(wins)
        result["groups"][name] = metrics
    pairwise = (placements["v2"][:, :, None] < placements["v1"][:, None, :]).mean(axis=(1, 2))
    delta = placements["v2"].mean(axis=1) - placements["v1"].mean(axis=1)
    result["v2_vs_v1"] = {
        "pairwise_ahead_rate": float(pairwise.mean()),
        "pairs_per_lobby": 9, "pair_comparisons": 9 * len(rows),
        "mean_placement_delta_v2_minus_v1": float(delta.mean()),
    }
    if confidence:
        result["v2_vs_v1"].update(pairwise_ahead_rate_ci95=interval(pairwise),
                                mean_placement_delta_ci95=interval(delta))
        result["confidence_method"] = "10,000 whole-lobby bootstrap samples; percentile 95% intervals"
    assert sum(group["first_places"] for group in result["groups"].values()) == len(rows)
    assert sum(sum(group["placement_counts"].values()) for group in result["groups"].values()) == len(rows) * 8
    return result


def main():
    config = json.loads((FOLDER / "config.json").read_text())
    assert config["games"] % config["parallel_games"] == 0
    assert config["lineup"] == {"scripted": 2, "v1": 3, "v2": 3}
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

    n_games, roles = config["parallel_games"], 6
    quota = config["games"] // n_games
    env = tg.BatchEnv(n_games, seed=config["seed"], controlled=range(roles),
                      threads=config["threads"], config=config["simulator_config"], rotate_seats=True)
    slots = np.arange(env.n_slots)
    lane = slots // roles
    # Native action execution follows role order. Balance which policy acts first.
    identity = ((slots % roles) + lane % 2) % 2
    identity = identity.reshape(n_games, roles)
    counts = np.zeros(n_games, dtype=np.int64)
    places = np.zeros((n_games, roles), dtype=np.int64)
    actions_taken = np.zeros_like(places)
    seats = env.slot_seats.reshape(n_games, roles).copy()
    deferred = 0
    rows = []
    started = time.monotonic()
    metadata = {"status": "running", "started_at": now(), "config": "config.json",
                "training_started": False, "optimizer_updates": 0, "weights_uploaded": False,
                "native_build": tg.build_info(), "models": config["models"]}
    write_json(FOLDER / "experiment.json", metadata)
    tracker = WandbTracker(FOLDER, config, kind="frozen-policy-lobby-match")
    tracker.run.define_metric("evaluated_lobbies")
    tracker.run.define_metric("test/*", step_metric="evaluated_lobbies")
    tracking = json.loads((FOLDER / "wandb.json").read_text())
    tracking["x_axis"] = "evaluated_lobbies"
    write_json(FOLDER / "wandb.json", tracking)
    last_report = time.monotonic()
    last_logged_count = 0
    batch_sizes = {name: [] for name in policies}
    exit_code = 1
    try:
        with (FOLDER / "games.jsonl").open("x", buffering=1) as stream:
            for vector_step in range(20000):
                active = counts < quota
                if not active.any():
                    break
                obs = env.observe()
                ready = obs["ready"] & active[lane]
                actions = np.zeros(env.n_slots, dtype=np.int64)
                for policy_id, name in enumerate(("v1", "v2")):
                    selected = np.flatnonzero(ready & (identity.reshape(-1) == policy_id))
                    if not len(selected):
                        continue
                    inputs = {k: v[selected] for k, v in obs.items() if k != "ready"}
                    with torch.inference_mode():
                        predicted, _ = policies[name].predict(inputs, deterministic=False)
                    assert inputs["action_mask"][np.arange(len(selected)), predicted].all()
                    actions[selected] = predicted
                    batch_sizes[name].append(len(selected))
                _, dones, info = env.step(actions, active=active)
                if info["rejected"].any() or ((info["placement"] == 0) & dones).any():
                    for i in np.unique(lane[info["rejected"] | dones]):
                        write_json(FOLDER / f"guard-game-{i}.json", json.loads(env.save(int(i))))
                    raise RuntimeError("Native rejection or terminal result without a real placement")
                deferred += int(info["deferred"].sum())
                actions_taken += (ready & ~info["deferred"]).reshape(n_games, roles)
                terminal = dones.reshape(n_games, roles)
                assert not (terminal & (places > 0)).any(), "Duplicate terminal report"
                places[terminal] = info["placement"].reshape(n_games, roles)[terminal]
                for i in np.flatnonzero(active & (places > 0).all(axis=1)):
                    assert len(set(places[i].tolist())) == roles
                    script_places = sorted(set(range(1, 9)) - set(places[i].tolist()))
                    assert len(script_places) == 2
                    # The native driver resets once all six model seats finish.
                    # Unique ranks 1..8 make the scripts' aggregate exact by complement;
                    # we do not pretend to know which individual script took each rank.
                    row = {"lane": int(i), "episode_index": int(counts[i]),
                           "placements": {name: places[i, identity[i] == j].tolist()
                                          for j, name in enumerate(("v1", "v2"))},
                           "model_seats": {name: seats[i, identity[i] == j].tolist()
                                           for j, name in enumerate(("v1", "v2"))},
                           "model_actions": {name: actions_taken[i, identity[i] == j].tolist()
                                             for j, name in enumerate(("v1", "v2"))},
                           "scripted_seats": sorted(set(range(8)) - set(seats[i].tolist()))}
                    row["placements"]["scripted"] = script_places
                    assert sorted(sum(row["placements"].values(), [])) == list(range(1, 9))
                    assert env.episode_index(int(i)) == int(counts[i]) + 1
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    counts[i] += 1
                    places[i] = actions_taken[i] = 0
                seats = env.slot_seats.reshape(n_games, roles).copy()
                if len(rows) >= last_logged_count + 64:
                    summary = summarize(rows)
                    tracker.log({"evaluated_lobbies": len(rows),
                                 **{f"test/{name}_{metric}": values[metric]
                                    for name, values in summary["groups"].items()
                                    for metric in ("first_rate", "top4_rate", "mean_placement", "lobby_win_share")}}, 0)
                    last_logged_count = len(rows)
                if time.monotonic() - last_report >= 15:
                    progress = {"status": "running", "updated_at": now(), "completed_lobbies": len(rows),
                                "target_lobbies": config["games"], "vector_steps": vector_step + 1,
                                "elapsed_seconds": time.monotonic() - started}
                    write_json(FOLDER / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    last_report = time.monotonic()
            else:
                raise RuntimeError("Match exceeded its bounded vector-step budget")
        assert len(rows) == config["games"] and (counts == quota).all()
        assert all(digest(ROOT / spec["path"]) == spec["sha256"] for spec in config["models"].values())
        result = {"status": "completed", "completed_at": now(), "config": config,
                  **summarize(rows, confidence=True), "elapsed_seconds": time.monotonic() - started,
                  "vector_steps": vector_step + 1, "deferred_actions": deferred, "rejected_actions": 0,
                  "weights_unchanged": True, "training_started": False, "optimizer_updates": 0,
                  "forward_batch_sizes": {name: {"calls": len(values), "mean": float(np.mean(values)),
                                                 "p50": float(np.median(values)), "max": max(values)}
                                          for name, values in batch_sizes.items()},
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "limitations": ["Results apply to this 2-script/3-V1/3-V2 stochastic BF16 lobby composition.",
                                  "Player seats in a lobby are correlated; confidence intervals resample whole lobbies.",
                                  "Script ranks are the exact unordered complement of the six unique model ranks.",
                                  "Pairwise ahead rate compares final placements, not individual combat outcomes."]}
        write_json(FOLDER / "results.json", result)
        tracker.log({"evaluated_lobbies": len(rows),
                     **{f"test/{name}_{metric}": values[metric]
                        for name, values in result["groups"].items()
                        for metric in ("first_rate", "top4_rate", "mean_placement", "lobby_win_share")},
                     **{f"test/v2_vs_v1_{key}": value for key, value in result["v2_vs_v1"].items()
                        if isinstance(value, (float, int))}}, 0)
        metadata.update(status="completed", completed_at=now(), results="results.json",
                        completed_lobbies=len(rows), elapsed_seconds=result["elapsed_seconds"])
        write_json(FOLDER / "experiment.json", metadata)
        write_json(FOLDER / "progress.json", metadata)
        print(json.dumps({"groups": result["groups"], "v2_vs_v1": result["v2_vs_v1"]}), flush=True)
        exit_code = 0
    except BaseException as exc:
        metadata.update(status="failed", failed_at=now(), error=f"{type(exc).__name__}: {exc}",
                        completed_lobbies=len(rows))
        write_json(FOLDER / "experiment.json", metadata)
        raise
    finally:
        tracker.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
