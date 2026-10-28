"""Train masked PPO against heuristic bots or a fixed pool of frozen policies.

Run from the repository root: python -m training.train --run-dir runs/ppo
No external model API is used. See docs/TRAINING.md for limits and provenance.
"""

from __future__ import annotations

import argparse
import copy
from collections import deque
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import taverngym as tg
from .common import (atomic_json, check_step, evaluate, generalized_advantage,
                     load_checkpoint, make_policy, stats, tensors)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--init-from", type=Path, help="Start a new stage from a trusted checkpoint")
    p.add_argument("--text-embeddings", type=Path)
    p.add_argument("--opponents", nargs="+", default=[],
                   help="Frozen checkpoint paths or bot; repeated entries weight the league pool")
    p.add_argument("--cancel-guard", action="store_true")
    p.add_argument("--opponent-batch", type=int, default=256)
    p.add_argument("--model", choices=("entity", "mlp"), default="entity")
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--threads", type=int, default=18)
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--embedding", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.00002)
    p.add_argument("--warmup-lr", type=float, default=0.0001)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--gae-lambda", type=float, default=0.995)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy", type=float, default=0.002)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--aux-coef", type=float, default=0.1)
    p.add_argument("--grad-norm", type=float, default=1.0)
    p.add_argument("--anchor-coef", type=float, default=0.1)
    p.add_argument("--anchor-updates", type=int, default=100)
    p.add_argument("--warmup-samples", type=int, default=262144)
    p.add_argument("--eval-games", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--eval-seed", type=int, default=800000001)
    p.add_argument("--action-cap", type=int, default=128)
    p.add_argument("--max-hours", type=float, default=24)
    p.add_argument("--max-updates", type=int, default=0)
    p.add_argument("--checkpoint-seconds", type=float, default=300)
    p.add_argument("--patience", type=int, default=30,
                   help="Stop after this many validation rounds without a new best; 0 disables")
    p.add_argument("--skip-baselines", action="store_true")
    args = p.parse_args()
    if args.resume and args.init_from:
        p.error("Choose exact resume or a new stage, not both")
    if args.init_from and args.warmup_samples:
        p.error("A new stage requires --warmup-samples 0")
    if args.opponents and args.warmup_samples:
        p.error("League training requires --warmup-samples 0")
    if args.cancel_guard and not args.opponents:
        p.error("History-aware training requires the league environment (--opponents)")
    if min(args.envs, args.rollout, args.minibatch, args.epochs, args.eval_every) < 1:
        p.error("Batch sizes, epochs and evaluation interval must be positive")
    if args.eval_games < 8 or args.eval_games % 8:
        p.error("--eval-games must be a positive multiple of eight")
    if args.seed == args.eval_seed:
        p.error("Training and validation seeds must differ")
    return args


def main():
    args = arguments()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "latest.pt").exists() and not args.resume:
        raise SystemExit("Existing run: specify --resume or use a new --run-dir")
    device = torch.device(args.device)
    torch.set_num_threads(args.torch_threads)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    config = {"turn_action_caps": [args.action_cap]}
    opponent_models, opponent_manifest = [], []
    if args.opponents:
        from .league import LeagueEnv, load_opponents
        opponent_models, opponent_manifest = load_opponents(args.opponents, device)
        if any(m.get("action_cap", args.action_cap) != args.action_cap for m in opponent_manifest):
            raise ValueError("A frozen opponent was trained with a different action cap")
        env = LeagueEnv(args.envs, seed=args.seed, opponents=opponent_models,
                        opponent_ids=[m["sha256"] for m in opponent_manifest], device=device,
                        threads=args.threads, config=config, inference_batch=args.opponent_batch,
                        cancel_guard=args.cancel_guard)
    else:
        env = tg.BatchEnv(args.envs, seed=args.seed, controlled=(0,), threads=args.threads,
                          config=config)
    architecture = (dict(kind="entity", width=args.width, layers=args.layers, heads=args.heads)
                    if args.model == "entity" else dict(kind="mlp", hidden=args.hidden, embedding=args.embedding))
    model = make_policy(architecture).to(device)
    anchor = None
    if args.init_from:
        model, _ = load_checkpoint(args.init_from, device)
        anchor = copy.deepcopy(model).eval().requires_grad_(False)
    elif args.text_embeddings and args.model == "entity" and not args.resume:
        text_data = torch.load(args.text_embeddings, map_location="cpu", weights_only=True)
        if text_data["engine_hash"] != tg.source_hash():
            raise ValueError("Text embeddings were generated for a different card table/build")
        model.text_vectors.copy_(text_data["vectors"].to(device))
    elif args.model == "entity" and not args.resume:
        raise ValueError("Prepare rule-text embeddings and pass --text-embeddings")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.warmup_lr if args.warmup_samples else args.lr, eps=1e-5)
    update = steps = warmup_done = 0
    best = float("inf")
    stale = 0
    previous_seconds = 0.0
    phase = "warmup" if args.warmup_samples else "ppo"
    if args.resume:
        model, ckpt = load_checkpoint(args.resume, device)
        if ckpt.get("anchor") is not None:
            anchor = copy.deepcopy(model).eval().requires_grad_(False)
            anchor.load_state_dict(ckpt["anchor"])
        for key in ("envs", "seed", "gamma", "gae_lambda", "action_cap"):
            if getattr(args, key) != ckpt["args"][key]:
                raise ValueError(f"Resume changes {key}; use the original setting")
        if bool(args.opponents) != bool(ckpt["args"].get("opponents")):
            raise ValueError("Resume changes environment type; use --init-from for a new stage")
        if args.cancel_guard != ckpt["args"].get("cancel_guard", False):
            raise ValueError("Resume changes cancellation decoding")
        if [m["sha256"] for m in opponent_manifest] != [m["sha256"] for m in ckpt.get("opponent_manifest", [])]:
            raise ValueError("Resume changes the frozen opponent pool")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
        optimizer.load_state_dict(ckpt["optimizer"])
        update, steps = ckpt["update"], ckpt["steps"]
        warmup_done, phase = ckpt["warmup_done"], ckpt["phase"]
        best, stale = ckpt["best_placement"], ckpt["stale_evals"]
        previous_seconds = ckpt["elapsed_seconds"]
        states = json.loads(gzip.decompress(ckpt["environment"]))
        if len(states) != args.envs:
            raise ValueError("Checkpoint environment count differs")
        for i, state in enumerate(states):
            env.load(i, state)
        torch.set_rng_state(ckpt["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
        np.random.set_state(ckpt["numpy_rng"])
    started = time.monotonic()
    sources = sorted(Path(__file__).parent.glob("*.py"))
    source_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sources)).hexdigest()
    snapshot = run / ("training_source_" + source_hash[:12])
    snapshot.mkdir(exist_ok=True)
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    stopped = False
    stop_reason = "running"
    last_checkpoint = started
    recent = deque(maxlen=4096)
    writer = SummaryWriter(str(run / "tensorboard"), purge_step=steps if args.resume else None)
    metrics = (run / "metrics.jsonl").open("a", buffering=1)
    (run / "pid").write_text(str(os.getpid()) + "\n")

    def elapsed():
        return previous_seconds + time.monotonic() - started

    def emit(event, **fields):
        row = {"time": datetime.now(timezone.utc).isoformat(), "event": event,
               "phase": phase, "update": update, "steps": steps,
               "elapsed_seconds": elapsed(), **fields}
        text = json.dumps(row, allow_nan=False)
        metrics.write(text + "\n")
        print(text, flush=True)
        atomic_json(run / "status.json", row)

    def save(name="latest.pt"):
        nonlocal last_checkpoint
        data = {
            "format_version": 1, "engine_hash": tg.source_hash(),
            "training_code_hash": source_hash,
            "architecture": model.architecture,
            "opponent_manifest": opponent_manifest,
            "anchor": anchor.state_dict() if anchor is not None else None,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "update": update, "steps": steps, "warmup_done": warmup_done, "phase": phase,
            "best_placement": best, "stale_evals": stale, "elapsed_seconds": elapsed(),
            "environment": gzip.compress(json.dumps([env.save(i) for i in range(args.envs)]).encode()),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        tmp = run / (name + ".tmp")
        torch.save(data, tmp)
        tmp.replace(run / name)
        last_checkpoint = time.monotonic()

    def stop_signal(signum, frame):
        nonlocal stopped, stop_reason
        stopped, stop_reason = True, f"signal_{signum}"

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_signal)

    def should_stop():
        nonlocal stop_reason
        if stopped:
            return True
        if (run / "STOP").exists():
            stop_reason = "stop_file"
            return True
        if args.max_hours and elapsed() >= args.max_hours * 3600:
            stop_reason = "time_budget"
            return True
        if args.max_updates and update >= args.max_updates:
            stop_reason = "update_budget"
            return True
        if args.patience and stale >= args.patience:
            stop_reason = "validation_plateau"
            return True
        return False

    def validation():
        nonlocal best, stale
        if opponent_models:
            from .arena import play_arena
            roles = [model] + [opponent_models[i % len(opponent_models)] for i in range(7)]
            arena = play_arena(roles, games=args.eval_games, seed=args.eval_seed,
                               threads=args.threads, config=config, device=device,
                               inference_batch=args.opponent_batch, cancel_guard=args.cancel_guard)
            arena.update(opponent_manifest=opponent_manifest, training_steps=steps,
                         training_code_hash=source_hash)
            report = run / f"league_eval_{update:06d}.json"
            atomic_json(report, arena)
            score = {**arena["candidate"], "seed": args.eval_seed, "kind": "league",
                     "opponents": "fixed frozen checkpoint pool and heuristic bots",
                     "rules": config, "report": str(report),
                     "guards_checked": arena["guards_checked"], "decoding": arena["decoding"]}
        else:
            score = evaluate(model, games=args.eval_games, seed=args.eval_seed,
                             threads=args.threads, config=config, device=device)
        improved = score["mean_placement"] < best
        stale = 0 if improved else stale + 1
        if improved:
            best = score["mean_placement"]
            save("best.pt")
            atomic_json(run / "best_eval.json", {"update": update, "steps": steps, **score})
        for key in ("mean_placement", "top4_rate", "win_rate"):
            writer.add_scalar("validation/" + key, score[key], steps)
        emit("evaluation", improved=improved, stale_evals=stale, **score)
        save()

    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "engine": tg.build_info(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "training_code_hash": source_hash, "architecture": model.architecture,
        "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "rules": config, "provenance": "Independent PPO baseline, not the paper's TavernAgent",
        "environment": "frozen_opponent_league" if args.opponents else "heuristic_bots",
        "opponent_manifest": opponent_manifest,
        "decoding": {"cancel_retry_guard": args.cancel_guard},
    }
    atomic_json(run / ("resume_manifest.json" if args.resume else "manifest.json"), manifest)
    emit("started", parameters=manifest["parameters"], device=str(device), resumed=bool(args.resume))
    try:
        if not args.resume:
            save()
            if not args.skip_baselines:
                for kind in ("random", "bot"):
                    score = evaluate(model, games=args.eval_games, seed=args.eval_seed,
                                     threads=args.threads, config=config, device=device, kind=kind)
                    atomic_json(run / (kind + "_baseline.json"), score)
                    emit("baseline", **score)

        # Supervised initialization learns sensible operations from the bundled weak bot.
        # Its outputs are not ground-truth strategy labels; PPO later optimizes placement.
        while warmup_done < args.warmup_samples and phase == "warmup" and not should_stop():
            fs, cs, ms, targets = [], [], [], []
            for _ in range(min(32, max(1, (args.warmup_samples - warmup_done + args.envs - 1) // args.envs))):
                obs = env.observe()
                f, c, m = tensors(obs, device)
                acts = np.array([env.game(i).bot_action(0) for i in range(args.envs)])
                fs.append(f); cs.append(c); ms.append(m)
                targets.append(torch.as_tensor(acts, device=device))
                _, done, info = env.step(acts)
                check_step(done, info)
                warmup_done += args.envs
            f, c, m, target = map(torch.cat, (fs, cs, ms, targets))
            losses, accuracy = [], []
            for _ in range(3):
                order = torch.randperm(len(target), device=device)
                for idx in order.split(args.minibatch):
                    logits, _ = model(f[idx], c[idx], m[idx])
                    loss = nn.functional.cross_entropy(logits, target[idx])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    losses.append(loss.item())
                    accuracy.append((logits.argmax(-1) == target[idx]).float().mean().item())
            emit("warmup", samples=warmup_done, loss=float(np.mean(losses)),
                 teacher_accuracy=float(np.mean(accuracy)))
            if time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                save()
        if phase == "warmup" and warmup_done >= args.warmup_samples:
            phase = "ppo"
            # Do not carry imitation-learning Adam moments into policy optimization.
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
            validation()
        elif not args.resume and not should_stop():
            validation()

        n, h = args.envs, args.rollout
        buffers = {
            "features": torch.empty((h, n, tg.OBS_FLOATS), device=device),
            "cards": torch.empty((h, n, tg.OBS_CARDS), dtype=torch.int32, device=device),
            "masks": torch.empty((h, n, tg.N_ACTIONS), dtype=torch.bool, device=device),
            "actions": torch.empty((h, n), dtype=torch.long, device=device),
            "logprobs": torch.empty((h, n), device=device),
            "values": torch.empty((h, n), device=device),
            "rewards": torch.empty((h, n), device=device),
            "dones": torch.empty((h, n), dtype=torch.bool, device=device),
            "placements": torch.empty((h, n), dtype=torch.long, device=device),
        }
        obs = env.observe()
        while phase == "ppo" and not should_stop():
            iteration_start = time.monotonic()
            for t in range(h):
                f, c, m = tensors(obs, device)
                with torch.no_grad():
                    dist, val = model.distribution(f, c, m)
                    action = dist.sample()
                    logprob = dist.log_prob(action)
                for key, x in (("features", f), ("cards", c), ("masks", m),
                               ("actions", action), ("logprobs", logprob), ("values", val)):
                    buffers[key][t].copy_(x)
                rewards, done, info = env.step(action.cpu().numpy())
                check_step(done, info)
                recent.extend(info["placement"][done].tolist())
                buffers["rewards"][t].copy_(torch.as_tensor(rewards, device=device))
                buffers["dones"][t].copy_(torch.as_tensor(done, device=device))
                buffers["placements"][t].copy_(torch.as_tensor(info["placement"].astype(np.int64), device=device))
                obs = env.observe()
                steps += n
            with torch.no_grad():
                _, bootstrap = model(*tensors(obs, device))
                advantages, returns = generalized_advantage(
                    buffers["rewards"], buffers["dones"], buffers["values"], bootstrap,
                    args.gamma, args.gae_lambda)
            b = {k: v.flatten(0, 1) for k, v in buffers.items()}
            adv, ret = advantages.flatten(), returns.flatten()
            labels = torch.full((h, n), -1, dtype=torch.long, device=device)
            future = torch.full((n,), -1, dtype=torch.long, device=device)
            for t in reversed(range(h)):
                future = torch.where(buffers["dones"][t], buffers["placements"][t] - 1, future)
                labels[t] = future
            labels = labels.flatten()
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            losses, kls, entropies, clips = [], [], [], []
            early_stop = False
            for epoch in range(args.epochs):
                epoch_kls = []
                for idx in torch.randperm(n * h, device=device).split(args.minibatch):
                    dist, value, place_logits = model.distribution(
                        b["features"][idx], b["cards"][idx], b["masks"][idx], auxiliary=True)
                    logratio = dist.log_prob(b["actions"][idx]) - b["logprobs"][idx]
                    ratio = logratio.exp()
                    kl = ((ratio.detach() - 1) - logratio.detach()).mean().item()
                    # Stop within an epoch. A full large rollout can contain many optimizer
                    # steps; checking only after the epoch lets the policy drift too far.
                    if kl > args.target_kl:
                        early_stop = True
                        break
                    pg = torch.maximum(-adv[idx] * ratio,
                                       -adv[idx] * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    vf = .5 * (value - ret[idx]).square().mean()
                    entropy = dist.entropy().mean()
                    valid_label = labels[idx] >= 0
                    auxiliary = nn.functional.cross_entropy(
                        place_logits[valid_label], labels[idx][valid_label]) if valid_label.any() else value.sum() * 0
                    anchored = value.sum() * 0
                    if anchor is not None and update < args.anchor_updates:
                        with torch.no_grad():
                            reference, _ = anchor.distribution(
                                b["features"][idx], b["cards"][idx], b["masks"][idx])
                        anchored = torch.distributions.kl_divergence(reference, dist).mean()
                    loss = (pg + args.value_coef * vf - args.entropy * entropy
                            + args.aux_coef * auxiliary + args.anchor_coef * anchored)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    epoch_kls.append(kl); kls.append(kl)
                    losses.append(loss.item()); entropies.append(entropy.item())
                    clips.append(((ratio.detach() - 1).abs() > args.clip).float().mean().item())
                if early_stop:
                    break
            if not losses:
                raise RuntimeError("KL limit reached before any optimizer step; check policy/rollout consistency")
            update += 1
            var = ret.var(unbiased=False)
            explained = (1 - (ret - b["values"]).var(unbiased=False) / var).item() if var > 1e-8 else 0.
            report = {
                "steps_per_second": n * h / (time.monotonic() - iteration_start),
                "loss": float(np.mean(losses)), "approx_kl": float(np.mean(kls)),
                "entropy": float(np.mean(entropies)), "clip_fraction": float(np.mean(clips)),
                "explained_variance": explained, "epochs": epoch + 1,
                "kl_early_stop": early_stop, "optimizer_steps": len(losses),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "recent_training": stats(recent), "rejected_actions": 0,
            }
            if args.opponents:
                report["opponent_decisions"] = sum(env.opponent_decisions)
            for key, value in report.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar("train/" + key, value, steps)
            emit("update", **report)
            if update % args.eval_every == 0 and not should_stop():
                validation()
            elif time.monotonic() - last_checkpoint >= args.checkpoint_seconds:
                save()
            writer.flush()
        save()
        emit("stopped", reason=stop_reason, checkpoint=str(run / "latest.pt"))
    except BaseException as exc:
        # Do not serialize partially updated optimizer state as an apparently good run.
        emit("failed", error=repr(exc), last_good_checkpoint=str(run / "latest.pt"))
        raise
    finally:
        writer.close()
        metrics.close()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
