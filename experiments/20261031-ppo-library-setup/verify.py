"""Installation smoke check; the PPO steps use a synthetic environment, not TavernGym."""

import argparse
import hashlib
import json
import platform
import sys
import tempfile
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import taverngym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("verification.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    assert sys.version_info[:2] == (3, 12), sys.version
    torch.set_num_threads(2)
    torch.manual_seed(0)

    cuda = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
    if args.require_cuda:
        assert cuda["available"], "CUDA is required for this check"
    if cuda["available"]:
        cuda.update(
            device=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)),
            compiled_architectures=torch.cuda.get_arch_list(),
        )
        x = torch.randn(64, 64, device="cuda", requires_grad=True)
        loss = (x @ x.T).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss).item()
        assert x.grad is not None and torch.isfinite(x.grad).all().item()
        cuda["matmul_and_backward"] = "passed"

    env = InvalidActionEnvDiscrete(dim=10, n_invalid_actions=5)
    model = MaskablePPO(
        "MlpPolicy", env, n_steps=16, batch_size=16, n_epochs=1,
        device="cpu", seed=0, verbose=0,
    )
    model.learn(total_timesteps=32)
    obs, _ = env.reset(seed=1)
    with tempfile.TemporaryDirectory(prefix="autotavern-ppo-smoke-") as tmp:
        model.save(Path(tmp) / "model")
        assert (Path(tmp) / "model.zip").is_file()
        restored = MaskablePPO.load(Path(tmp) / "model.zip", env=env, device="cpu")
        expected, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        actual, _ = restored.predict(obs, action_masks=env.action_masks(), deterministic=True)
        assert np.array_equal(expected, actual)
        for _ in range(32):
            mask = env.action_masks()
            action, _ = restored.predict(obs, action_masks=mask, deterministic=False)
            assert mask[int(action)]
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                obs, _ = env.reset()
    env.close()

    assert taverngym.build_is_current() is True
    result = {
        "status": "passed",
        "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {name: version(name) for name in (
            "torch", "stable-baselines3", "sb3-contrib", "gymnasium",
            "tensorboard", "taverngym", "numpy", "pytest",
        )},
        "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("pyproject.toml", "uv.lock", ".python-version")},
        "cuda": cuda,
        "maskable_ppo": {
            "environment": "sb3_contrib.common.envs.InvalidActionEnvDiscrete",
            "device": "cpu", "training_timesteps": model.num_timesteps,
            "masked_predictions_checked": 32,
            "save_load_prediction_match": True,
            "model_format": "SB3 native .zip; temporary smoke model removed",
            "game_training": False,
        },
        "taverngym": {
            "build": taverngym.build_info(),
            "build_is_current": True,
            "actions": taverngym.N_ACTIONS,
            "numeric_features": taverngym.OBS_FLOATS,
            "card_indices": taverngym.OBS_CARDS,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
