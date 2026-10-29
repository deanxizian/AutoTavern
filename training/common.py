"""Masked policy, return computation, and reproducible evaluation for TavernGym."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import taverngym as tg


class Policy(nn.Module):
    def __init__(self, hidden=768, embedding=32):
        super().__init__()
        self.hidden, self.embedding = hidden, embedding
        self.architecture = dict(kind="mlp", hidden=hidden, embedding=embedding)
        self.cards = nn.Embedding(tg.N_CARDS + 1, embedding, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(tg.OBS_FLOATS + tg.OBS_CARDS * embedding, hidden),
            nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.actor = nn.Linear(hidden, tg.N_ACTIONS)
        self.critic = nn.Linear(hidden, 1)
        self.placement = nn.Linear(hidden, 8)
        nn.init.normal_(self.actor.weight, std=0.01 / math.sqrt(hidden))
        nn.init.zeros_(self.actor.bias)
        nn.init.zeros_(self.critic.bias)

    def forward_aux(self, features, cards, masks):
        # Numeric observations already have units/scales, but late-game stats can grow
        # without bound. A fixed, invertible signed log needs no changing running stats.
        numeric = features.sign() * torch.log1p(features.abs())
        encoded = self.encoder(torch.cat((numeric, self.cards(cards.long()).flatten(1)), 1))
        logits = self.actor(encoded).masked_fill(~masks, -1e9)
        return logits, self.critic(encoded).squeeze(-1), self.placement(encoded)

    def forward(self, features, cards, masks):
        logits, value, _ = self.forward_aux(features, cards, masks)
        return logits, value

    def distribution(self, features, cards, masks, auxiliary=False):
        logits, value, placement = self.forward_aux(features, cards, masks)
        result = (Categorical(logits=logits, validate_args=False), value)
        return (*result, placement) if auxiliary else result


def tensors(obs, device):
    if not obs["ready"].all() or not obs["action_mask"].any(axis=1).all():
        raise RuntimeError("This baseline requires exactly one ready learning seat per game")
    if not np.isfinite(obs["features"]).all():
        raise RuntimeError("Simulator returned non-finite observations")
    return (
        torch.as_tensor(obs["features"], device=device),
        torch.as_tensor(obs["cards"], device=device),
        torch.as_tensor(obs["action_mask"], device=device),
    )


def generalized_advantage(rewards, dones, values, next_value, gamma, lam):
    """dones[t] terminates transition t, including when next obs is an auto-reset."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(next_value)
    for t in reversed(range(len(rewards))):
        continuation = (~dones[t]).float()
        following = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * following * continuation - values[t]
        carry = delta + gamma * lam * continuation * carry
        advantages[t] = carry
    return advantages, advantages + values


def check_step(dones, info, *, allow_truncation=False):
    if info["rejected"].any():
        raise RuntimeError("A masked action was rejected by the simulator")
    truncated = info.get("truncated", np.zeros_like(dones))
    if truncated.dtype != np.bool_ or truncated.shape != dones.shape:
        raise RuntimeError("Invalid training truncation flags")
    if truncated.any() and not allow_truncation:
        raise RuntimeError("A training truncation is not a completed evaluation episode")
    if np.any(truncated & (~dones | (info["placement"] != 0))):
        raise RuntimeError("A training truncation must reset without assigning a placement")
    if np.any(dones & ~truncated & ((info["placement"] < 1) | (info["placement"] > 8))):
        raise RuntimeError("Simulator stopped without a valid placement; refusing to train on it")


@torch.no_grad()
def bootstrap_time_limits(model, rewards, info, gamma, device):
    """Bootstrap truncated transitions from their final view, never the reset game.

    dones stays true for GAE so neither values nor advantages cross a reset. Adding
    gamma * V(final_observation) to the zero transition reward supplies exactly the
    missing one-step bootstrap. Truncations keep placement 0 (no auxiliary label).
    """
    lanes = np.flatnonzero(info.get("truncated", np.zeros(len(rewards), bool)))
    if not len(lanes):
        return rewards
    if not np.array_equal(lanes, info.get("truncated_lanes")) or np.any(rewards[lanes] != 0):
        raise RuntimeError("Truncation bootstrap lanes or rewards are inconsistent")
    obs = info.get("final_observation")
    if obs is None or len(obs["ready"]) != len(lanes):
        raise RuntimeError("Truncation is missing its final observation")
    logits, values = model(*tensors(obs, device))
    if not torch.isfinite(logits).all() or not torch.isfinite(values).all():
        raise RuntimeError("Non-finite truncation bootstrap")
    if values.shape != (len(lanes),):
        raise RuntimeError("Truncation value shape differs from its lanes")
    rewards = rewards.copy()
    rewards[lanes] += np.float32(gamma) * values.float().cpu().numpy()
    return rewards


def stats(places):
    a = np.asarray(places, dtype=np.float64)
    if not len(a):
        return {"games": 0}
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return {
        "games": len(a), "mean_placement": float(a.mean()),
        "placement_ci95": [float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se)],
        "top4_rate": float((a <= 4).mean()), "win_rate": float((a == 1).mean()),
        "placement_counts": [int((a == i).sum()) for i in range(1, 9)],
    }


@torch.no_grad()
def evaluate(model, *, games, seed, threads, config, device, kind="policy",
             audit_guards=False, profile=False, cancel_guard=False):
    """Fixed first episode per lane, equal games at all seats; never fastest-N sampling.

    These seeds are validation seeds when used to select a checkpoint, not a test set.
    Greedy policy evaluation always applies the same action mask as training.
    """
    if games < 8 or games % 8:
        raise ValueError("Evaluation games must be a positive multiple of eight")
    was_training = model.training
    model.eval()
    if getattr(model, "decoder_metadata", {}) and kind != "policy":
        raise ValueError("A position decoder requires policy evaluation")
    if hasattr(model, "reset_context"):
        model.reset_context()
    places, per_seat, raw = [], {}, {}
    from .diagnostics import BehaviorAudit
    behavior = BehaviorAudit() if profile else None
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for seat in range(8):
            from .decoding import CancelRetryGuard
            decoder = CancelRetryGuard() if cancel_guard and kind != "bot" else None
            env = tg.BatchEnv(games // 8, seed=seed + seat, controlled=(seat,),
                              threads=threads, config=config)
            first = np.zeros(env.n_games, dtype=np.uint8)
            # Auto-reset games need new identities even when their visible turn
            # repeats. IDs below `games` identify exactly the counted episodes.
            identities = seat * env.n_games + np.arange(env.n_games)
            for _ in range(30_000):
                obs = env.observe()
                policy_obs = decoder.apply(obs) if decoder is not None else obs
                before = ({i: env.game(i) for i in np.flatnonzero(first == 0)}
                          if audit_guards else {})
                if kind == "bot":
                    actions = np.array([env.game(i).bot_action(seat) for i in range(env.n_games)])
                else:
                    f, c, m = tensors(policy_obs, device)
                    if kind == "random":
                        logits = torch.zeros(m.shape, device=device).masked_fill(~m, -1e9)
                    else:
                        if hasattr(model, "set_context"):
                            model.set_context(identities)
                        logits, value = model(f, c, m)
                        if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                            raise RuntimeError("Evaluation policy produced non-finite outputs")
                    selected = (Categorical(logits=logits).sample() if kind == "random"
                                else logits.argmax(-1))
                    if not m.gather(1, selected[:, None]).all():
                        raise RuntimeError("Evaluation policy chose a masked action")
                    actions = selected.cpu().numpy()
                if behavior is not None:
                    behavior.record(obs, actions, np.flatnonzero(first == 0), seat * env.n_games)
                if decoder is not None:
                    decoder.record(actions)
                _, done, info = env.step(actions)
                check_step(done, info)
                take = done & (first == 0)
                if audit_guards:
                    from .diagnostics import replay_terminal_step
                    for i in np.flatnonzero(take):
                        replay_terminal_step(before[i], [actions[i]], (seat,), [done[i]],
                                             [info["placement"][i]],
                                             context={"seed": seed + seat, "lane": int(i), "seat": seat})
                first[take] = info["placement"][take]
                if decoder is not None:
                    for i in np.flatnonzero(done):
                        decoder.reset(i)
                if hasattr(model, "reset_context"):
                    for i in np.flatnonzero(done):
                        model.reset_context(int(identities[i]))
                    identities[done] += games
                if first.all():
                    break
            else:
                raise RuntimeError("Evaluation failed to finish within its driver limit")
            per_seat[str(seat)] = stats(first)
            raw[str(seat)] = first.tolist()
            places.extend(first.tolist())
    model.train(was_training)
    position_events = ([e for e in model.move_events if e["identity"] < games]
                       if getattr(model, "decoder_metadata", {}) else None)
    return {**stats(places), "seed": seed, "kind": kind, "per_seat": per_seat,
            "opponents": "seven bundled heuristic bots", "rules": config,
            "placements_by_seat": raw, "guards_checked": audit_guards,
            "truncated_combats": 0 if audit_guards else None,
            "turn_limit_ends": 0 if audit_guards else None,
            "behavior": behavior.report() if behavior is not None else None,
            **({"position_decoder": {"counted_episode_moves": len(position_events),
                                       "events": position_events}}
               if position_events is not None else {}),
            "decoding": {"kind": "greedy" if kind == "policy" else kind,
                         "cancel_retry_guard": bool(cancel_guard and kind != "bot"),
                         "guard_version": CancelRetryGuard.version if cancel_guard else None,
                         **getattr(model, "decoder_metadata", {})}}


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    # Only load trusted checkpoints produced by this trainer: optimizer/RNG state uses pickle.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt["engine_hash"] != tg.source_hash():
        raise ValueError("Checkpoint engine/card table differs from this checkout")
    model = make_policy(ckpt["architecture"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def make_policy(architecture):
    config = dict(architecture)
    kind = config.pop("kind", "mlp")
    if kind == "entity":
        from .entity import EntityPolicy
        return EntityPolicy(**config)
    return Policy(**config)
