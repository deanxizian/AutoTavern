"""Evaluate a trusted checkpoint on a fresh seed range, without changing training."""

import argparse
import hashlib
from pathlib import Path
import torch

from .common import atomic_json, evaluate
from .arena import frozen_checkpoint
from .diagnostics import SimulatorGuardError, check_seed_range


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
    if args.out.exists() or args.out.with_suffix(".failed.json").exists():
        p.error("Use a new output path; preserve earlier evaluation evidence")
    model, metadata = frozen_checkpoint(args.checkpoint, device)
    check_seed_range(args.seed, 8, metadata["train_seed"], metadata["validation_seed"])
    metadata["evaluation_code_hash"] = hashlib.sha256(b"".join(
        f.name.encode() + f.read_bytes() for f in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        score = evaluate(model, games=args.games, seed=args.seed, threads=args.threads,
                         config={"turn_action_caps": [metadata["action_cap"]]}, device=device,
                         audit_guards=True, profile=True)
    except SimulatorGuardError as exc:
        atomic_json(args.out.with_suffix(".failed.json"),
                    {"checkpoint": metadata, "passed": False, "guard": exc.evidence})
        raise
    score.update(checkpoint=metadata, training_steps=metadata["steps"])
    atomic_json(args.out, score)
    print(score)


if __name__ == "__main__":
    main()
