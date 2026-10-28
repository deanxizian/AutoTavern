"""Evaluate a trusted checkpoint on a fresh seed range, without changing training."""

import argparse
from pathlib import Path
import torch

from .common import atomic_json, evaluate, load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--seed", type=int, default=900000001)
    p.add_argument("--threads", type=int, default=18)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    model, ckpt = load_checkpoint(args.checkpoint, device)
    if args.seed in (ckpt["args"]["seed"], ckpt["args"]["eval_seed"]):
        p.error("Use a fresh seed range for this independent test")
    score = evaluate(model, games=args.games, seed=args.seed, threads=args.threads,
                     config={"turn_action_caps": [ckpt["args"]["action_cap"]]}, device=device)
    score.update(checkpoint=str(args.checkpoint), training_steps=ckpt["steps"])
    atomic_json(args.out, score)
    print(score)


if __name__ == "__main__":
    main()
