"""Measure learner recruit-turn tails above 60 with opponents still capped at 60."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
import torch
import taverngym as tg

from taverngym.learning.experiment import load_frozen_policy, load_frozen_opponents, now, write_json
from taverngym.learning.rewards import ActionRewardShaper, RewardShapingConfig
from taverngym.learning.vector_env import NativeVectorEnv
from taverngym.learning.tracking import WandbTracker

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
FOLDER = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/20261031-exp03-v2-continuation/20261031-ppo-v3-2v1-3v2-warmup"
GAMES, CAP, SEED = 64, 200, 61000301


def play(policy, opponents, config, mode):
    torch.manual_seed(SEED)
    roles = 6
    batch = tg.BatchEnv(GAMES, seed=SEED, controlled=range(roles), threads=4,
                        config=config["simulator_config"], rotate_seats=True)
    for lane in range(GAMES):
        state = json.loads(batch.save(lane))
        seat = int(batch.slot_seats[lane * roles])
        state["game"]["players"][seat]["uncapped"] = True
        assert all(not p["uncapped"] for i, p in enumerate(state["game"]["players"]) if i != seat)
        batch.load(lane, json.dumps(state))
    finished = np.zeros(GAMES, bool)
    shapers = [ActionRewardShaper(RewardShapingConfig(0, 0)) for _ in range(GAMES)]
    turns, episodes = [], []
    current = [None] * GAMES
    last_report = time.monotonic()
    slots = np.arange(batch.n_slots)
    groups = [(group, np.arange(1 + sum(g.count for g in opponents.groups[:i]),
                               1 + sum(g.count for g in opponents.groups[:i + 1])))
              for i, group in enumerate(opponents.groups)]

    def flush(lane, reason):
        if current[lane] is not None:
            turns.append({**current[lane], "close_reason": reason})
            current[lane] = None

    for vector_step in range(25000):
        if finished.all():
            break
        full = batch.observe()
        ready = full["ready"] & ~finished[slots // roles]
        learner_slots = np.flatnonzero(ready & (slots % roles == 0))
        actions = np.zeros(batch.n_slots, np.int64)
        forced = np.zeros(GAMES, bool)
        if len(learner_slots):
            inputs = {k: v[learner_slots].copy() for k, v in full.items() if k != "ready"}
            for row, slot in enumerate(learner_slots):
                lane = int(slot // roles)
                turn = int(round(float(inputs["features"][row, 0]) * 20))
                if turn < 1:
                    continue
                if current[lane] is None or current[lane]["turn"] != turn:
                    flush(lane, "next_turn")
                    current[lane] = dict(mode=mode, lane=lane, turn=turn, actions=0,
                        after60=0, non_end_after60=0, cycles_after60=0, moves_after60=0,
                        choices_after60=0, forced_end_at200=0, end_with_alternatives=0)
                if current[lane]["actions"] >= CAP and inputs["action_mask"][row, 0]:
                    inputs["action_mask"][row] = False
                    inputs["action_mask"][row, 0] = True
                    forced[lane] = True
            with torch.inference_mode():
                selected, _ = policy.predict(inputs, deterministic=mode == "greedy")
            assert inputs["action_mask"][np.arange(len(selected)), selected].all()
            actions[learner_slots] = selected
        for group, role_ids in groups:
            indices = np.flatnonzero(ready & np.isin(slots % roles, role_ids))
            if len(indices):
                inputs = {k: v[indices] for k, v in full.items() if k != "ready"}
                with torch.inference_mode():
                    selected, _ = group.policy.predict(inputs, deterministic=False)
                assert inputs["action_mask"][np.arange(len(selected)), selected].all()
                actions[indices] = selected
        _, dones, info = batch.step(actions, active=~finished)
        NativeVectorEnv._validate_result(dones, info)
        assert not info["deferred"][::roles].any()
        after = batch.observe()
        for slot in learner_slots:
            lane = int(slot // roles)
            shaper = shapers[lane]
            previous = shaper._totals["cycle_returns"]
            _, components = shaper.step(int(actions[slot]),
                {k: v[slot] for k, v in full.items() if k != "ready"},
                {k: v[slot] for k, v in after.items() if k != "ready"}, 0, done=bool(dones[slot]))
            cycles = (components or shaper._totals)["cycle_returns"] - previous
            row = current[lane]
            if row is not None:
                action = int(actions[slot])
                row["actions"] += 1
                if row["actions"] > 60:
                    row["after60"] += 1
                    row["non_end_after60"] += int(action != 0)
                    row["cycles_after60"] += int(cycles)
                    row["moves_after60"] += int(action == 2 or action >= 196)
                    row["choices_after60"] += int(30 <= action < 46 or action == 7)
                row["forced_end_at200"] += int(forced[lane] and action == 0)
                row["end_with_alternatives"] += int(action == 0 and not forced[lane]
                                                    and full["action_mask"][slot].sum() > 1)
        for lane in np.flatnonzero(dones[::roles] & ~finished):
            episodes.append(dict(lane=int(lane), placement=int(info["placement"][lane * roles])))
            flush(int(lane), "episode_end")
            finished[lane] = True
        if time.monotonic() - last_report > 15:
            print(mode, "completed", len(episodes), "turns", len(turns), "steps", vector_step, flush=True)
            last_report = time.monotonic()
    else:
        raise RuntimeError("Probe exhausted its bounded vector-step budget")
    assert len(episodes) == GAMES and all(row is None for row in current)
    return turns, episodes


def summary(turns):
    values = np.array([r["actions"] for r in turns])
    selected = [r for r in turns if r["actions"] > 60]
    tails = np.array([r["after60"] for r in selected])
    total_tail = sum(r["after60"] for r in turns)
    cycles = sum(r["cycles_after60"] for r in turns)
    continued = [r for r in turns if r["non_end_after60"] > 0]
    extra = np.array([r["non_end_after60"] for r in continued])
    return {"turns": len(turns), "turns_over60": len(selected),
        "fraction_over60": len(selected) / len(turns), "mean_actions": float(values.mean()),
        "turns_continuing_after60": len(continued), "fraction_continuing_after60": len(continued) / len(turns),
        "mean_extra_non_end_if_continued": float(extra.mean()) if len(extra) else 0,
        "extra_non_end_p50_p90_p95_p99": np.quantile(extra, [.5, .9, .95, .99]).tolist() if len(extra) else [],
        "actions_p50_p90_p95_p99": np.quantile(values, [.5, .9, .95, .99]).tolist(),
        "max_actions": int(values.max()), "mean_tail_if_over60": float(tails.mean()) if len(tails) else 0,
        "tail_p50_p90_p95_p99": np.quantile(tails, [.5, .9, .95, .99]).tolist() if len(tails) else [],
        "total_actions_after60": total_tail, "non_end_actions_after60": sum(r["non_end_after60"] for r in turns),
        "move_freeze_after60": sum(r["moves_after60"] for r in turns),
        "cycles_after60": cycles, "cycle_fraction_after60": cycles / max(total_tail, 1),
        "forced_end_at200_turns": sum(r["forced_end_at200"] > 0 for r in turns),
        "forced_end_at200_fraction": float(np.mean([r["forced_end_at200"] > 0 for r in turns])),
        "end_with_alternatives_turns": sum(r["end_with_alternatives"] > 0 for r in turns)}


def main():
    if (FOLDER / "results.json").exists():
        raise FileExistsError("Refusing to overwrite a completed probe")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    config = json.loads((SOURCE / "resolved-config.json").read_text())
    raw = (ROOT / "runs" / SOURCE.name / "latest_model.zip").read_bytes()
    stage = json.loads(zipfile.ZipFile(io.BytesIO(raw)).read("data"))["num_timesteps"] // 131072
    frozen = load_frozen_opponents(config, "cuda")
    frozen.set_forward_precision("bf16")
    started = time.monotonic()
    meta = {"name": FOLDER.name, "checkpoint_ppo_iteration": stage,
        "snapshot_sha256": hashlib.sha256(raw).hexdigest(), "games_per_mode": GAMES,
        "opponents": "2 fixed V1 + 3 fixed V2 + 2 scripted, all remain capped at 60",
        "learner_diagnostic_cap": CAP, "learner_forward_precision": "bf16",
        "training_started": False, "weights_uploaded": False,
        "modes": ["sampled", "greedy"], "seed": SEED}
    write_json(FOLDER / "config.json", meta)
    with tempfile.NamedTemporaryFile(suffix=".zip", prefix="autotavern-turn-tail-") as file:
        file.write(raw)
        file.flush()
        policy = load_frozen_policy(file.name, "cuda", forward_precision="bf16")
        results = {}
        for mode in meta["modes"]:
            turns, episodes = play(policy, frozen, config, mode)
            with (FOLDER / f"{mode}-turns.jsonl").open("x") as stream:
                for row in turns:
                    stream.write(json.dumps(row) + "\n")
            write_json(FOLDER / f"{mode}-episodes.json", episodes)
            results[mode] = summary(turns)
            print(mode, json.dumps(results[mode]), flush=True)
    write_json(FOLDER / "results.json", {**meta, "status": "completed", "completed_at": now(),
        "elapsed_seconds": time.monotonic() - started, "conditions": results,
        "limitations": ["Only the learner budget is relaxed; no training or action-loop filtering.",
            "200 is a safety threshold; turns hitting it have unknown uncapped lengths.",
            "Counts include EndTurn and pending choices; episode-ending partial turns are retained.",
            "This observes behavior after relaxing the cap, not the benefit of the extra actions."]})
    tracker = WandbTracker(FOLDER, meta, kind="turn-actions-over60")
    try:
        tracker.log({f"test/{mode}_{key}": value for mode, row in results.items()
                     for key, value in row.items() if isinstance(value, (int, float))}, 0)
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
