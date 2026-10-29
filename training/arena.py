"""Seat-balanced matches against frozen checkpoints and the bundled heuristic.

Every reported row is the first complete lobby in its lane. No player or lobby is
selected by how quickly it finishes. This is simulator evaluation, not human MMR.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

import taverngym as tg
from .common import atomic_json, check_step, load_checkpoint, stats, tensors
from .diagnostics import BehaviorAudit, SimulatorGuardError, check_seed_range, replay_terminal_step
from .decoding import CancelRetryGuard
from .turns import RoundRobinArena
from .diagnostics import check_game_guards


def wilson(successes, total):
    """Two-sided 95% Wilson interval for one Bernoulli result per independent lobby."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("An interval needs valid success and lobby counts")
    z = 1.959963984540054
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [0. if successes == 0 else max(0., float(center - half)),
            1. if successes == total else min(1., float(center + half))]


def placement_stats(places):
    places = np.asarray(places)
    return {**stats(places),
            "top4_ci95": wilson(int((places <= 4).sum()), len(places)),
            "win_ci95": wilson(int((places == 1).sum()), len(places))}


def rotations(games):
    if games < 8 or games % 8:
        raise ValueError("Arena games must be a positive multiple of eight")
    # Role zero is the candidate. Each role occupies each seat equally often.
    return (np.arange(8)[None, :] - np.arange(games)[:, None] % 8) % 8


@torch.inference_mode()
def play_arena(policies, *, games, seed, threads, config, device,
               inference_batch=256, max_steps=240_000, env_factory=RoundRobinArena,
               audit_guards=True, cancel_guard=False):
    """Eight role policies: None means the bundled bot; otherwise a torch policy.

    Waiting and eliminated seats are never passed to a policy. The default driver
    retains terminal games. Alternative drivers may reset; later episodes never count.
    """
    if len(policies) != 8 or inference_batch < 1:
        raise ValueError("Supply eight role policies and a positive inference batch")
    roles = rotations(games)
    flat_roles = roles.reshape(-1)
    env = env_factory(games, seed=seed, controlled=tuple(range(8)), threads=threads,
                      config=config)
    first = np.zeros((games, 8), dtype=np.uint8)
    decision_counts = np.zeros(8, dtype=np.int64)
    behavior = BehaviorAudit()
    decoder = CancelRetryGuard() if cancel_guard else None
    learned = np.array([policy is not None for policy in policies])[flat_roles]
    started = time.monotonic()
    # Repeated opponents share one forward pass and one loaded model.
    groups = {}
    for role, policy in enumerate(policies):
        groups.setdefault(id(policy), (policy, []))[1].append(role)
    modes = [(policy, policy.training) for policy, _ in groups.values() if policy is not None]
    for policy, _ in modes:
        policy.eval()
    try:
        for driver_step in range(max_steps):
            active_lanes = ~first.all(axis=1)
            obs = env.observe()
            retained = getattr(env, "terminal_is_retained", False)
            before = ({i: env.game(i) for i in np.flatnonzero(active_lanes)}
                      if audit_guards and not retained else {})
            ready = obs["ready"] & np.repeat(active_lanes, 8)
            policy_obs = (decoder.apply({**obs, "ready": ready & learned})
                          if decoder is not None else obs)
            actions = np.zeros(env.n_slots, dtype=np.int64)
            for policy, members in groups.values():
                slots = np.flatnonzero(ready & np.isin(flat_roles, members))
                if not len(slots):
                    continue
                decision_counts += np.bincount(flat_roles[slots], minlength=8)
                if policy is None:
                    copies = {}
                    for slot in slots:
                        lane, seat = divmod(int(slot), 8)
                        if lane not in copies:
                            copies[lane] = env.game(lane)
                        actions[slot] = copies[lane].bot_action(seat)
                else:
                    for offset in range(0, len(slots), inference_batch):
                        subset = slots[offset:offset + inference_batch]
                        inputs = tensors({k: v[subset] for k, v in policy_obs.items()}, device)
                        logits, value = policy(*inputs)
                        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                            raise RuntimeError("Arena policy produced non-finite outputs")
                        chosen = logits.argmax(-1)
                        if not inputs[2].gather(1, chosen[:, None]).all():
                            raise RuntimeError("Arena policy chose a masked action")
                        actions[subset] = chosen.cpu().numpy()
            behavior.record(obs, actions, np.flatnonzero(ready & (flat_roles == 0)))
            if decoder is not None:
                slots = np.flatnonzero(ready & learned)
                decoder.record(actions[slots], identities=slots)
            _, done, info = env.step(actions)
            relevant = np.repeat(active_lanes, 8)
            check_step(done[relevant], {k: v[relevant] for k, v in info.items()})
            take = done & relevant & (first.reshape(-1) == 0)
            if audit_guards:
                for lane in np.flatnonzero(take.reshape(games, 8).any(axis=1)):
                    slots = slice(lane * 8, (lane + 1) * 8)
                    context = {"seed": seed, "lane": int(lane), "driver_step": driver_step}
                    if retained:
                        terminal = env.game(lane)
                        if list(terminal.placements()) != info["placement"][slots].tolist():
                            raise RuntimeError("Arena placement report differs from retained game")
                        check_game_guards(terminal, context=context)
                    else:
                        replay_terminal_step(before[lane], actions[slots], tuple(range(8)),
                                             done[slots], info["placement"][slots], context=context)
            first.reshape(-1)[take] = info["placement"][take]
            if decoder is not None:
                for slot in np.flatnonzero(done):
                    decoder.reset(slot)
            if first.all():
                break
        else:
            raise RuntimeError("Arena failed to finish within its driver limit")
    finally:
        for policy, training in modes:
            policy.train(training)
        if hasattr(env, "close"):
            env.close()
    if not np.all(np.sort(first, axis=1) == np.arange(1, 9)):
        raise RuntimeError("A completed lobby does not contain placements 1 through 8")
    by_role = np.take_along_axis(first, roles.argsort(axis=1), axis=1)
    candidate = by_role[:, 0]
    opponents = {}
    for role in range(1, 8):
        wins = int((candidate < by_role[:, role]).sum())
        opponents[str(role)] = {
            **placement_stats(by_role[:, role]),
            "candidate_beats_role_rate": wins / games,
            "candidate_beats_role_ci95": wilson(wins, games),
        }
    return {
        "candidate": placement_stats(candidate), "opponents": opponents,
        "per_candidate_seat": {
            str(seat): placement_stats(candidate[np.arange(games) % 8 == seat])
            for seat in range(8)
        },
        "seed": seed, "rules": config, "games": games,
        "driver": getattr(env, "driver", "batch_snapshot"),
        "driver_steps": driver_step + 1,
        "decisions_per_role": decision_counts.tolist(),
        "elapsed_seconds": time.monotonic() - started,
        "rejected_actions": 0, "incomplete_lobbies": 0,
        "guards_checked": audit_guards,
        "truncated_combats": 0 if audit_guards else None,
        "turn_limit_ends": 0 if audit_guards else None,
        "candidate_behavior": behavior.report(),
        "decoding": {"kind": "greedy", "cancel_retry_guard": cancel_guard,
                     "guard_version": CancelRetryGuard.version if cancel_guard else None,
                     "scope": "all learned policies equally; heuristic unchanged"},
        "lobbies": [{"lane": lane, "role_by_seat": roles[lane].tolist(),
                     "placement_by_seat": first[lane].tolist()}
                    for lane in range(games)],
    }


def frozen_checkpoint(path, device):
    # Hash and load the same open file, even if the trainer atomically replaces best.pt.
    # Full training checkpoints are trusted pickle; never load untrusted downloads.
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        handle.seek(0)
        model, checkpoint = load_checkpoint(handle, "cpu")
    metadata = {"path": str(path.resolve()), "sha256": digest.hexdigest(),
                "steps": checkpoint["steps"], "update": checkpoint["update"],
                "engine_hash": checkpoint["engine_hash"],
                "training_code_hash": checkpoint["training_code_hash"],
                "train_seed": checkpoint["args"]["seed"],
                "validation_seed": checkpoint["args"]["eval_seed"],
                "action_cap": checkpoint["args"]["action_cap"]}
    if checkpoint.get("weight_transfer"):
        metadata["weight_transfer"] = checkpoint["weight_transfer"]
    if checkpoint.get("initial_policy"):
        metadata["initial_policy"] = checkpoint["initial_policy"]
    metadata["trained_hotfix_37_0_1"] = checkpoint["args"].get("hotfix_37_0_1", False)
    metadata["training_turn_limit"] = checkpoint["args"].get("league_turn_limit", 0)
    metadata["truncated_training_episodes"] = checkpoint.get("truncated_training_episodes", 0)
    del checkpoint
    return model.to(device), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", help="Trusted checkpoint, or bot for a harness check")
    parser.add_argument("--opponents", nargs=7, default=["bot"] * 7,
                        help="Seven trusted checkpoint paths or the literal bot")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cancel-guard", action="store_true")
    parser.add_argument("--hotfix-37-0-1", action=argparse.BooleanOptionalAction, default=None,
                        help="Defaults to the candidate's training profile; applies to every seat")
    parser.add_argument("--inference-batch", type=int, default=256)
    parser.add_argument("--action-cap", type=int, default=128)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".failed.json").exists():
        parser.error("Use a new output path; preserve earlier evaluation evidence")
    if tg.build_is_current() is not True:
        parser.error("Build the native extension from the current checkout first")
    rotations(args.games)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    source_hash = hashlib.sha256(b"".join(
        p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    cache, policies, members = {}, [], []
    for name in [args.candidate, *args.opponents]:
        key = "bot" if name == "bot" else str(Path(name).resolve())
        if key not in cache:
            cache[key] = ((None, {"kind": "bundled_heuristic"}) if name == "bot"
                          else frozen_checkpoint(Path(name), device))
        model, metadata = cache[key]
        if model is not None:
            check_seed_range(args.seed, 1, metadata["train_seed"], metadata["validation_seed"])
            if args.action_cap != metadata["action_cap"]:
                parser.error("Arena action cap differs from a checkpoint's training rules")
        policies.append(model)
        members.append(metadata)
    hotfix = (members[0].get("trained_hotfix_37_0_1", False) if args.hotfix_37_0_1 is None
              else args.hotfix_37_0_1)
    metadata = dict(format_version=3, role_models=members, engine=tg.build_info(),
                    game_version=tg.game_version(), torch_version=torch.__version__,
                    evaluation_code_hash=source_hash,
                    protocol="one first episode per lane; one candidate per lobby; seat-ordered player views")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = play_arena(policies, games=args.games, seed=args.seed, threads=args.threads,
                            config={"turn_action_caps": [args.action_cap],
                                    **({"hotfix_37_0_1": True} if hotfix else {})}, device=device,
                            inference_batch=args.inference_batch, cancel_guard=args.cancel_guard)
    except Exception as exc:
        failure = {**metadata, "passed": False, "error": repr(exc),
                   "seed": args.seed, "games": args.games}
        if isinstance(exc, SimulatorGuardError):
            failure["guard"] = exc.evidence
        atomic_json(args.out.with_suffix(".failed.json"),
                    failure)
        raise
    result.update(metadata)
    atomic_json(args.out, result)
    print({"out": str(args.out), "candidate": result["candidate"],
           "elapsed_seconds": result["elapsed_seconds"]}, flush=True)


if __name__ == "__main__":
    main()
