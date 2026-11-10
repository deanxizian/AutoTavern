"""Observe reversible-action loops in the frozen model; never change its policy."""

from collections import Counter
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
import taverngym as tg
from taverngym.learning.ppo import EntityPPO
from taverngym.learning.vector_env import NativeVectorEnv


FOLDER = Path(__file__).resolve().parent
ROOT = next(p for p in FOLDER.parents if (p / "pyproject.toml").is_file())
MODEL = ROOT / "runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip"
CONFIG = {"turn_action_caps": [60]}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def keys(obs, i, layout):
    features, cards, mask = obs["features"][i], obs["cards"][i], obs["action_mask"][i]
    visible = digest(features.tobytes() + cards.tobytes())
    board = digest(features[layout.floats["board"][0]].tobytes() + cards[layout.cards["board"]].tobytes())
    return visible, board, digest(visible.encode() + mask.tobytes())


def own(batch):
    return json.loads(batch.game(0).view(0))["me"]


def board_ids(me):
    return [m["eid"] for m in me["board"]]


def shop_state(me):
    return [{key: card.get(key) for key in ("eid", "hid", "card", "frozen")} for card in me["shop"]]


def save_case(mode, kind, segment, loop_start, episode, cases):
    if sum(c["mode"] == mode and c["kind"] == kind for c in cases) >= 3:
        return
    replay = tg.BatchEnv(1, seed=0, controlled=(0,), config=CONFIG)
    replay.load(0, segment["snapshot"])
    states = [own(replay)]
    for action in segment["actions"]:
        _, done, info = replay.step([action])
        assert not done[0] and not info["rejected"][0]
        states.append(own(replay))
    exact = (board_ids(states[loop_start]) == board_ids(states[-1]) if kind == "move_return" else
             shop_state(states[loop_start]) == shop_state(states[-1]))
    if not exact:
        return
    name = f"{mode}-{kind}-{1 + sum(c['mode'] == mode and c['kind'] == kind for c in cases):02d}"
    folder = FOLDER / "cases"
    folder.mkdir(exist_ok=True)
    path = folder / (name + ".state.json")
    path.write_text(segment["snapshot"] + "\n")
    case = {"mode": mode, "kind": kind, "episode": episode, "turn": segment["turn"],
            "snapshot": str(path.relative_to(FOLDER)), "actions": segment["actions"].copy(),
            "action_names": [tg.action_name(a) for a in segment["actions"]],
            "loop_start": loop_start, "loop_end": len(segment["actions"]),
            "exact_native_replay_verified": True,
            "trace": [{"step": j, "board": [{k: m.get(k) for k in ("eid", "card", "name", "atk", "hp")} for m in me["board"]],
                       "board_eids": board_ids(me), "frozen": me["frozen"], "shop": shop_state(me)}
                      for j, me in enumerate(states)]}
    cases.append(case)


def run_mode(model, deterministic, cases):
    mode = "greedy" if deterministic else "sampled"
    torch.manual_seed(50000001)
    env = NativeVectorEnv(32, seed=50000001, threads=4, config=CONFIG)
    obs = env.reset()
    counts = np.zeros(32, dtype=int)
    episodes = [Counter() for _ in range(32)]
    segments, turns = [None] * 32, [None] * 32
    game_rows, totals = [], Counter()
    layout = model.policy.network.layout
    started = time.perf_counter()
    with gzip.open(FOLDER / f"{mode}-actions.jsonl.gz", "xt") as stream:
        while (counts < 2).any():
            actions, _ = model.predict(obs, deterministic=deterministic)
            before = {}
            for i in np.flatnonzero(counts < 2):
                a = int(actions[i])
                turn = round(float(obs["features"][i, 0]) * 20)
                episode = int(env.batch.episode_index(int(i)))
                if turns[i] != turn:
                    turns[i] = turn
                    episodes[i]["turn_actions"] = 0
                    segments[i] = None
                    if turn:
                        episodes[i]["turns"] += 1
                if turn and episodes[i]["turn_actions"] >= 60 and a == 0:
                    assert obs["action_mask"][i].sum() == 1
                    episodes[i]["forced_end_at_cap"] += 1
                episodes[i]["actions"] += 1
                episodes[i]["turn_actions"] += 1
                reversible = a == 2 or a >= 196
                old_keys = keys(obs, i, layout)
                if reversible and segments[i] is None:
                    segments[i] = {"turn": turn, "snapshot": env.batch.save(int(i)), "actions": [],
                                   "visible_seen": {old_keys[0]: 0}, "board_seen": {old_keys[1]: 0},
                                   "input_seen": {old_keys[2]: 0}}
                before[i] = {"action": a, "turn": turn, "episode": episode, "keys": old_keys,
                             "frozen": bool(obs["features"][i, 15]), "reversible": reversible}
            new_obs, _, dones, infos = env.step(actions)
            for i, previous in before.items():
                a, turn = previous["action"], previous["turn"]
                current_turn = round(float(new_obs["features"][i, 0]) * 20)
                same_turn = not dones[i] and turn == current_turn
                events = []
                if same_turn and previous["reversible"]:
                    segment = segments[i]
                    segment["actions"].append(a)
                    n = len(segment["actions"])
                    episodes[i]["max_reversible_streak"] = max(episodes[i]["max_reversible_streak"], n)
                    after = keys(new_obs, i, layout)
                    if after[0] in segment["visible_seen"]:
                        episodes[i]["visible_state_returns"] += 1
                        events.append("visible_state_return")
                    if after[2] in segment["input_seen"]:
                        episodes[i]["identical_policy_input_returns"] += 1
                    if a >= 196:
                        episodes[i]["moves"] += 1
                        if after[1] == previous["keys"][1]:
                            episodes[i]["moves_without_visible_change"] += 1
                        elif after[1] in segment["board_seen"]:
                            episodes[i]["move_board_returns"] += 1
                            events.append("move_return")
                            save_case(mode, "move_return", segment, segment["board_seen"][after[1]],
                                      {"slot": int(i), "episode_index": previous["episode"]}, cases)
                        if n > 1 and segment["actions"][-2] >= 196:
                            src, dst = divmod(a - 196, 7)
                            old_src, old_dst = divmod(segment["actions"][-2] - 196, 7)
                            if (src, dst) == (old_dst, old_src):
                                episodes[i]["immediate_inverse_moves"] += 1
                    else:
                        episodes[i]["freeze_presses"] += 1
                        if n > 1 and segment["actions"][-2] == 2:
                            episodes[i]["consecutive_freeze_pairs"] += 1
                        if after[0] in segment["visible_seen"]:
                            episodes[i]["freeze_state_returns"] += 1
                            events.append("freeze_return")
                            save_case(mode, "freeze_return", segment, segment["visible_seen"][after[0]],
                                      {"slot": int(i), "episode_index": previous["episode"]}, cases)
                    for name, key in zip(("visible_seen", "board_seen", "input_seen"), after):
                        segment[name][key] = n
                else:
                    segments[i] = None
                stream.write(json.dumps({"slot": int(i), "episode": previous["episode"], "turn": turn,
                                         "step": episodes[i]["actions"], "action": a,
                                         "name": tg.action_name(a), "frozen_before": previous["frozen"],
                                         "frozen_after": bool(new_obs["features"][i, 15]) if same_turn else None,
                                         "events": events}) + "\n")
                if dones[i]:
                    record = {"slot": int(i), "episode": previous["episode"],
                              "placement": infos[i]["placement"], **episodes[i]}
                    record.pop("turn_actions", None)
                    game_rows.append(record)
                    totals.update({k: v for k, v in record.items() if k not in {"slot", "episode", "placement", "max_reversible_streak"}})
                    episodes[i] = Counter()
                    segments[i], turns[i] = None, None
                    counts[i] += 1
            obs = new_obs
    env.close()
    placements = [r["placement"] for r in game_rows]
    result = {"mode": mode, "games": len(game_rows), "elapsed_seconds": time.perf_counter() - started,
              "mean_placement": float(np.mean(placements)), "first_rate": placements.count(1) / 64,
              "top4_rate": sum(p <= 4 for p in placements) / 64, "totals": dict(totals),
              "games_with_move_returns": sum(r.get("move_board_returns", 0) > 0 for r in game_rows),
              "games_with_freeze_returns": sum(r.get("freeze_state_returns", 0) > 0 for r in game_rows),
              "games_with_cap_end": sum(r.get("forced_end_at_cap", 0) > 0 for r in game_rows),
              "max_reversible_streak": max(r.get("max_reversible_streak", 0) for r in game_rows),
              "per_game": game_rows}
    write(FOLDER / f"{mode}-results.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "per_game"}, ensure_ascii=False), flush=True)
    return result


def main():
    if (FOLDER / "results.json").exists():
        raise FileExistsError("Preserve the completed diagnostic; choose a new output directory")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    before = digest(MODEL.read_bytes())
    model = EntityPPO.load(MODEL, device="cuda", n_envs=1, n_steps=2, batch_size=2)
    steps = model.optimizer_step_count()
    cases = []
    results = {"greedy": run_mode(model, True, cases), "sampled": run_mode(model, False, cases)}
    write(FOLDER / "cases.json", cases)
    assert digest(MODEL.read_bytes()) == before and model.optimizer_step_count() == steps
    record = {"model": str(MODEL.relative_to(ROOT)), "model_sha256": before,
              "seed": 50000001, "n_envs": 32, "games_per_mode": 64,
              "simulator_config": CONFIG, "opponents": "seven identical scripted bots",
              "policy_modified": False, "model_weights_unchanged": True,
              "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "modes": results, "native_replay_verified_cases": len(cases),
              "counting": "state-return counts include overlapping returns within an uninterrupted move/freeze segment; never across turns or productive actions"}
    write(FOLDER / "results.json", record)


if __name__ == "__main__":
    main()
