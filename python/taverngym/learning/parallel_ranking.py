"""Finite development ranking: all candidate lobbies live, one batch per weight."""

from collections import defaultdict
from dataclasses import asdict
import json
import time

import numpy as np
import torch

from .experiment import EVALUATION_METRICS, load_frozen_policy, now, write_json
from .rewards import ActionRewardShaper, RewardShapingConfig, reward_protocols_equal
from .stage_chain import digest, inspect_checkpoint, require


KEYS = ("features", "cards", "action_mask")


def one(obs, slot):
    return {key: obs[key][slot] for key in KEYS}


def merged_actions(shards, policies, statistics):
    """Merge candidate and reference seats across every shard by checkpoint hash."""
    for shard in shards:
        shard["obs"] = shard["env"].observe()
        roles = shard.get("role_count", shard["env"].n_slots // len(shard["active"]))
        shard["ready"] = shard["obs"]["ready"] & np.repeat(shard["active"], roles)
        shard["actions"] = np.zeros(shard["env"].n_slots, dtype=np.int64)
    for identity, policy in policies.items():
        chunks = [(s, np.flatnonzero(s["ready"] & (s["identities"] == identity))) for s in shards]
        chunks = [(s, ids) for s, ids in chunks if len(ids)]
        if not chunks:
            continue
        obs = {key: np.concatenate([s["obs"][key][ids] for s, ids in chunks]) for key in KEYS}
        with torch.inference_mode():
            actions, _ = policy.predict(obs, deterministic=False)
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        require(obs["action_mask"][np.arange(len(actions)), actions].all(),
                "Ranking policy emitted an illegal action")
        offset = 0
        for shard, ids in chunks:
            shard["actions"][ids] = actions[offset:offset + len(ids)]
            offset += len(ids)
        statistics[identity].append([len(actions), len(chunks)])


def parallel_evaluate(root, folder, inputs, checkpoints, device):
    import taverngym as tg

    protocol = inputs["protocol"]
    games, seed = protocol["games"], protocol["seed"]
    candidates = inputs["candidates"]
    reference = protocol["reference_pool"]
    frozen = sum(s["count"] for s in reference)
    role_count = 1 + frozen
    require(games == 64 and frozen == (7 if inputs.get("pool_selection") == "top2" else 6),
            "Unexpected parallel ranking protocol")
    require(not (folder / "ranking-games.jsonl").exists(),
            "Partial parallel ranking exists; preserve it and diagnose before rerunning")
    all_specs = [*candidates, *reference]
    policies, versions = {}, {}
    for spec in all_specs:
        path = root / spec["model"]
        require(digest(path) == spec["sha256"], "Ranking model hash changed before loading")
        checkpoint = checkpoints.setdefault(spec["model_version"], inspect_checkpoint(path))
        require(checkpoint["sha256"] == spec["sha256"], "Conflicting ranking model version")
        actual = checkpoint["reward_shaping"]
        # Legacy V1-V3 specs deliberately carry null: their saved metadata is
        # authoritative. Newly trained generations require an exact protocol.
        if spec["model_version"] in ("v1", "v2", "v3"):
            require(not actual or (actual.get("cycle_forfeit_repetitions") is None
                                  and actual.get("turn_action_forfeit_limit") is None),
                    "Legacy ranking model has unexpected forfeits")
        else:
            require(checkpoint["model_version"] == spec["model_version"]
                    and reward_protocols_equal(actual, spec.get("reward_shaping")),
                    "Ranking checkpoint version or protocol differs from sealed specification")
        require(not actual or (actual.get("cycle_forfeit_repetitions") is None
                              and actual.get("turn_action_forfeit_limit") is None),
                "Parallel native ranking requires disabled early forfeits")
    if inputs.get("pool_selection") == "top2":
        fingerprints = [checkpoints[s["model_version"]]["policy_sha256"] for s in candidates]
        require(len(set(fingerprints)) == len(fingerprints),
                "Duplicate candidate policies must be registered as retired aliases before ranking")
    for spec in all_specs:
        path = root / spec["model"]
        require(digest(path) == spec["sha256"], "Ranking model hash changed before loading")
        if spec["sha256"] not in policies:
            policies[spec["sha256"]] = load_frozen_policy(path, device, forward_precision="bf16")
        versions[spec["model_version"]] = spec["sha256"]
    shards = []
    # Each shard receives the same native seeds and seat rotation; all shards
    # exist before any policy inference. No lane is reset or replaced.
    for spec in candidates:
        roles = [spec, *(s for s in reference for _ in range(s["count"]))]
        configs = [checkpoints[r["model_version"]]["reward_shaping"] for r in roles]
        require(all(not c or (c.get("cycle_forfeit_repetitions") is None
                             and c.get("turn_action_forfeit_limit") is None) for c in configs),
                "Parallel native ranking requires disabled early forfeits")
        env = tg.BatchEnv(games, seed=seed, controlled=range(role_count), threads=4,
                          config={"turn_action_caps": [60]}, rotate_seats=True)
        actual = [ActionRewardShaper(RewardShapingConfig(**(configs[i % role_count] or {
                      "operation_cost": 0, "cycle_cost": 0})))
                  for i in range(games * role_count)]
        diagnostic = [ActionRewardShaper(RewardShapingConfig(
            operation_cost=0, cycle_cost=.01, cycle_detection="all_actions"))
            for _ in range(games * role_count)]
        shards.append({"spec": spec, "roles": roles, "configs": configs, "env": env,
                       "initial_seats": env.slot_seats.copy(), "role_count": role_count,
                       "identities": np.tile([s["sha256"] for s in roles], games),
                       "active": np.ones(games, dtype=bool),
                       "places": np.zeros((games, role_count), dtype=np.int64),
                       "actual": actual, "diagnostic": diagnostic,
                       "turns": [defaultdict(lambda: {"actions": 0, "forced_ends": 0,
                           "cycle_returns": 0, "max_consecutive_cycle_returns": 0})
                           for _ in range(games * role_count)],
                       "totals": [None] * (games * role_count), "episodes": [None] * games})
    statistics = defaultdict(list)
    requested = games * len(candidates)
    started = time.monotonic()
    metadata = {"started_at": now(), "requested_concurrency": requested,
                "actual_initial_concurrency": sum(len(s["active"]) for s in shards),
                "games_per_candidate": games, "seed": seed, "precision": "bf16",
                "automatic_terminal_resets": 0, "replacement_games": 0, "hash_to_versions": versions,
                "controlled_seats_per_game": role_count,
                "v0_seats_per_game": 8 - role_count,
                "diagnostics": ("all_actions; read-only; V0 behavior unknown" if role_count == 7
                                else "all_actions; read-only; all eight seats controlled")}
    write_json(folder / "ranking-scheduler.json", metadata)
    rounds = deferred = 0
    inference_seconds = environment_seconds = 0.
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        with (folder / "ranking-games.jsonl").open("x") as stream:
            while any(s["active"].any() for s in shards):
                tick = time.monotonic()
                merged_actions(shards, policies, statistics)
                inference_seconds += time.monotonic() - tick
                rounds += 1
                for s in shards:
                    if not s["active"].any():
                        continue
                    tick = time.monotonic()
                    rewards, dones, info = s["env"].step(s["actions"], active=s["active"])
                    environment_seconds += time.monotonic() - tick
                    require(not info["rejected"].any(), "Native ranking action rejected")
                    require(not ((info["placement"] == 0) & dones).any(),
                            "Ranking terminal missing real placement")
                    deferred += int(info["deferred"].sum())
                    after = s["env"].observe()
                    for slot in np.flatnonzero(s["ready"] & ~info["deferred"]):
                        before_one, after_one = one(s["obs"], slot), one(after, slot)
                        turn = int(round(float(before_one["features"][0]) * 20))
                        row = s["turns"][slot][turn]
                        forced = (turn > 0 and s["actions"][slot] == 0 and row["actions"] >= 60
                                  and before_one["action_mask"].sum() == 1)
                        old = s["diagnostic"][slot]._totals["cycle_returns"]
                        _, diag = s["diagnostic"][slot].step(s["actions"][slot], before_one,
                            after_one, float(rewards[slot]), done=bool(dones[slot]))
                        total = diag or s["diagnostic"][slot]._totals
                        row["actions"] += 1
                        row["forced_ends"] += int(forced)
                        row["cycle_returns"] += int(total["cycle_returns"] - old)
                        row["max_consecutive_cycle_returns"] = max(
                            row["max_consecutive_cycle_returns"],
                            int(s["diagnostic"][slot].consecutive_returns))
                        _, actual = s["actual"][slot].step(s["actions"][slot], before_one,
                            after_one, float(rewards[slot]), done=bool(dones[slot]))
                        if actual is not None:
                            s["totals"][slot] = actual
                    for slot in np.flatnonzero(dones):
                        lane, role = divmod(int(slot), role_count)
                        require(s["places"][lane, role] == 0, "Duplicate ranking terminal")
                        place = int(info["placement"][slot])
                        s["places"][lane, role] = place
                        if s["totals"][slot] is None:
                            s["totals"][slot] = s["actual"][slot].finish_terminal(
                                float(rewards[slot]), place)
                        if role == 0:
                            turns = s["turns"][slot]
                            recruit = [r for turn, r in turns.items() if turn > 0]
                            components = dict(s["totals"][slot])
                            components["cap_end_rate"] = (sum(r["forced_ends"] for r in recruit)
                                                           / max(len(recruit), 1))
                            s["episodes"][lane] = {"placement": place,
                                "turn_before_terminal_action": max(turns, default=0),
                                "episode": {"r": components["total_reward"],
                                            "l": int(components["actions_per_game"])},
                                "termination_reason": "native_terminal",
                                "reward_components": components,
                                "learner_seat": int(s["initial_seats"][slot])}
                    finished = s["active"] & (s["places"] > 0).all(axis=1)
                    for lane in np.flatnonzero(finished):
                        places = s["places"][lane].tolist()
                        v0 = sorted(set(range(1, 9)) - set(places))
                        require(len(v0) == 8 - role_count and len(set(places)) == role_count,
                                "Invalid eight-seat ranking")
                        players = [{"model_version": spec["model_version"],
                                    "checkpoint_sha256": spec["sha256"],
                                    "seat": int(s["initial_seats"][lane * role_count + role]),
                                    "placement": places[role],
                                    "turns": dict(s["turns"][lane * role_count + role])}
                                   for role, spec in enumerate(s["roles"])]
                        record = {"candidate": s["spec"]["model_version"], "lane": int(lane),
                                  "seed": seed, "episode_index": 0, "players": players,
                                  "v0_placements": v0, "v0_behavior": None}
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        require(s["env"].episode_index(int(lane)) == 1,
                                "Ranking lane was unexpectedly reset")
                        s["active"][lane] = False
    finally:
        for s in shards:
            if hasattr(s["env"], "close"):
                s["env"].close()
    results = {}
    for s in shards:
        episodes = s["episodes"]
        require(all(episodes), "Missing ranking candidate terminal")
        placements = [e["placement"] for e in episodes]
        result = {"games": games, "seed": seed, "deterministic": False,
                  "placements": placements, "episodes": episodes,
                  "mean_placement": sum(placements) / games,
                  "first_rate": placements.count(1) / games,
                  "top4_rate": sum(p <= 4 for p in placements) / games,
                  "opponent_groups": [{"name": r["name"], "count": r["count"],
                      "deterministic": False, "reward_shaping": r.get("reward_shaping")}
                      for r in reference]}
        result.update({name: float(np.mean([e["reward_components"][name] for e in episodes]))
                       for name in EVALUATION_METRICS})
        result["reward_shaping"] = (asdict(RewardShapingConfig(**s["configs"][0]))
                                    if s["configs"][0] is not None else None)
        results[s["spec"]["model_version"]] = result
    require(all(digest(root / spec["model"]) == spec["sha256"] for spec in all_specs),
            "Ranking model changed during inference")
    metadata.update(completed_at=now(), completed_games=requested, rounds=rounds,
                    automatic_terminal_resets=requested,
                    deferred_actions=deferred, rejected_actions=0,
                    elapsed_seconds=time.monotonic() - started,
                    games_per_second=requested / (time.monotonic() - started),
                    inference_seconds=inference_seconds, environment_seconds=environment_seconds,
                    peak_cuda_allocated_bytes=(torch.cuda.max_memory_allocated()
                                               if str(device).startswith("cuda") else None),
                    batch_distribution=dict(statistics), hashes_verified_after=True,
                    games_sha256=digest(folder / "ranking-games.jsonl"))
    write_json(folder / "ranking-scheduler.json", metadata)
    return results
