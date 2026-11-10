"""Read-only inference traces of the original V3 under the stopped V4 rules."""

import json
from pathlib import Path
import types

import numpy as np
import torch
import taverngym as tg

from taverngym.learning.experiment import load_frozen_opponents, now, write_json
from taverngym.learning.ppo import EntityPPO, PlacementBuffer
from taverngym.learning.precision import configure_precision
from taverngym.learning.rewards import ActionRewardShaper
from taverngym.learning.vector_env import MixedOpponentVectorEnv


def main():
    folder = Path(__file__).resolve().parent
    root = next(path for path in folder.parents if (path / "pyproject.toml").is_file())
    config = json.loads((root / "experiments/20261031-exp04-v4-cycle-only/"
                         "20261031-ppo-v4-cycle-only/config.json").read_text())
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    source = root / config["initial_model"]
    model = EntityPPO.load(source, device="cuda", n_envs=1, n_steps=2, batch_size=2,
                          rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
    opponents = load_frozen_opponents(config, "cuda")
    configure_precision(model, config, opponents)
    games = []
    snapshots = {}
    for deterministic in (False, True):
        for seed in (44, 45, 46):
            torch.manual_seed(seed)
            env = MixedOpponentVectorEnv(1, seed=seed, threads=1,
                opponent_policy=opponents, frozen_opponents=5,
                config=config["simulator_config"], reward_shaping=config["reward_shaping"])
            obs = env.reset()
            shaper = env._reward_shapers[0]
            original_step = shaper.step
            trace = []

            def record_step(self, action, before, after, native_reward, **kwargs):
                old_key = ActionRewardShaper._visible(before).hex()
                key = ActionRewardShaper._visible(after).hex()
                for state, value in ((old_key, before), (key, after)):
                    if state not in snapshots:
                        snapshots[state] = {k: value[k].tolist() for k in ("features", "cards")}
                previous_cycles = self.consecutive_cycles
                result = original_step(action, before, after, native_reward, **kwargs)
                trace.append({"index": len(trace) + 1, "action": int(action),
                    "name": tg.action_name(int(action)), "before_sha256": old_key,
                    "after_sha256": key, "unchanged_visible_state": old_key == key,
                    "turn_before": round(float(before["features"][0]) * 20),
                    "turn_after": round(float(after["features"][0]) * 20),
                    "previous_cycles": previous_cycles, "cycles_after": self.consecutive_cycles,
                    "forfeit_requested": self.forfeit_requested,
                    "native_reward_before_forfeit": float(native_reward),
                    "native_done_before_forfeit": kwargs.get("done"),
                    "action_was_legal": bool(before["action_mask"][int(action)]),
                    "changed_features": np.flatnonzero(before["features"] != after["features"]).tolist(),
                    "changed_cards": np.flatnonzero(before["cards"] != after["cards"]).tolist()})
                return result

            shaper.step = types.MethodType(record_step, shaper)
            for _ in range(500):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, infos = env.step(action)
                if done[0]:
                    games.append({"seed": seed, "learner_mode": "greedy" if deterministic else "sampled",
                        "opponents": "2 fixed V1 + 3 fixed V2 + 2 scripted; model opponents sampled",
                        "info": infos[0], "terminal_transition_reward": float(reward[0]), "trace": trace})
                    break
            else:
                games.append({"seed": seed, "learner_mode": "greedy" if deterministic else "sampled",
                              "stopped_at_trace_limit": True, "trace": trace})
            env.close()
    output = {"created_at": now(), "purpose": "development diagnosis only; no optimizer updates",
              "source_model": str(source), "source_sha256": config["initial_model_sha256"],
              "seed_protocol": "training seeds 44/45/46; independent heldout not read",
              "reward_shaping": config["reward_shaping"], "games": games,
              "visible_state_snapshots": snapshots}
    write_json(folder / "initial-cycle-traces.json", output)
    for game in games:
        print(json.dumps({"mode": game["learner_mode"], "seed": game["seed"],
            "placement": game.get("info", {}).get("placement"),
            "reason": game.get("info", {}).get("termination_reason"),
            "actions": [{"i": row["index"], "action": row["name"],
                         "cycles": row["cycles_after"], "same": row["unchanged_visible_state"],
                         "before": row["before_sha256"][:8], "after": row["after_sha256"][:8]}
                        for row in game["trace"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
