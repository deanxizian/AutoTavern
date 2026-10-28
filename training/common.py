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


def check_step(dones, info):
    if info["rejected"].any():
        raise RuntimeError("A masked action was rejected by the simulator")
    if np.any(dones & ((info["placement"] < 1) | (info["placement"] > 8))):
        raise RuntimeError("Simulator stopped without a valid placement; refusing to train on it")


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
def evaluate(model, *, games, seed, threads, config, device, kind="policy"):
    """Fixed first episode per lane, equal games at all seats; never fastest-N sampling.

    These seeds are validation seeds when used to select a checkpoint, not a test set.
    Greedy policy evaluation always applies the same action mask as training.
    """
    if games < 8 or games % 8:
        raise ValueError("Evaluation games must be a positive multiple of eight")
    was_training = model.training
    model.eval()
    places, per_seat = [], {}
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        for seat in range(8):
            env = tg.BatchEnv(games // 8, seed=seed + seat, controlled=(seat,),
                              threads=threads, config=config)
            first = np.zeros(env.n_games, dtype=np.uint8)
            for _ in range(30_000):
                obs = env.observe()
                if kind == "bot":
                    actions = np.array([env.game(i).bot_action(seat) for i in range(env.n_games)])
                else:
                    f, c, m = tensors(obs, device)
                    if kind == "random":
                        logits = torch.zeros(m.shape, device=device).masked_fill(~m, -1e9)
                    else:
                        logits, _ = model(f, c, m)
                    actions = (Categorical(logits=logits).sample() if kind == "random"
                               else logits.argmax(-1)).cpu().numpy()
                _, done, info = env.step(actions)
                check_step(done, info)
                take = done & (first == 0)
                first[take] = info["placement"][take]
                if first.all():
                    break
            else:
                raise RuntimeError("Evaluation failed to finish within its driver limit")
            per_seat[str(seat)] = stats(first)
            places.extend(first.tolist())
    model.train(was_training)
    return {**stats(places), "seed": seed, "kind": kind, "per_seat": per_seat,
            "opponents": "seven bundled heuristic bots", "rules": config}


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
