"""Append recorded evaluation curves to TensorBoard without reevaluating models."""

import hashlib
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main():
    folder = Path(__file__).resolve().parent
    paths = sorted(folder.glob("evaluation-[0-9][0-9][0-9][0-9].json"))
    paths += [folder / f"evaluation-{name}-heldout.json" for name in ("initial", "best", "final")]
    fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    marker = folder / "tensorboard" / "evaluation-export.json"
    if marker.exists() and json.loads(marker.read_text()) == fingerprints:
        print("Evaluation curves already exported.")
        return
    with SummaryWriter(str(folder), filename_suffix=".evaluation") as writer:
        for path in paths:
            result = json.loads(path.read_text())
            heldout = "heldout" in path.name
            if heldout:
                step = result.get("model_timesteps", 819200)
            else:
                step = int(path.stem.split("-")[-1]) * 8192
            for name in ("mean_placement", "first_rate", "top4_rate"):
                writer.add_scalar(f"{'heldout' if heldout else 'eval'}/{name}", result[name], step)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(fingerprints, indent=2) + "\n")
    print("Exported fixed-seed and held-out placement/first/top4 curves.")


if __name__ == "__main__":
    main()
