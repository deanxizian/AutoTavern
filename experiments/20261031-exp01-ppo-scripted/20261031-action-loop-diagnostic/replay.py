"""Bounded greedy replay of a captured loop; --uncapped changes only this copy."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import taverngym as tg
from taverngym.learning.ppo import EntityPPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--uncapped", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.steps <= 512:
        raise ValueError("Choose 1..512 steps for a bounded diagnostic")
    folder = Path(__file__).resolve().parent
    root = next(p for p in folder.parents if (p / "pyproject.toml").is_file())
    case = next(c for c in json.loads((folder / "cases.json").read_text())
                if c["mode"] == "greedy" and c["kind"] == "move_return")
    snapshot = json.loads((folder / case["snapshot"]).read_text())
    if args.uncapped:
        snapshot["game"]["cfg"]["turn_action_caps"] = []
    env = tg.BatchEnv(1, seed=0, controlled=(0,))
    env.load(0, json.dumps(snapshot))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    model = EntityPPO.load(root / "runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip",
                           device="cuda", n_envs=1, n_steps=2, batch_size=2)
    actions, boards, inputs = [], [], []
    for _ in range(args.steps):
        obs = env.observe()
        obs.pop("ready")
        inputs.append(hashlib.sha256(b"".join(obs[k].tobytes() for k in
                                              ("features", "cards", "action_mask"))).hexdigest())
        boards.append([m["eid"] for m in json.loads(env.game(0).view(0))["me"]["board"]])
        action = int(model.predict(obs, deterministic=True)[0][0])
        actions.append(action)
        _, _, info = env.step([action])
        assert not info["rejected"][0]
    record = {"mode": "greedy", "steps": args.steps, "source_case": case["snapshot"],
              "changed_environment_setting": {"turn_action_caps": []} if args.uncapped else {},
              "policy_modified": False, "all_actions_are_moves": all(a >= 196 for a in actions),
              "unique_policy_input_states": len(set(inputs)),
              "unique_board_orders": len({tuple(board) for board in boards}), "actions": actions,
              "first_action_names": [tg.action_name(a) for a in actions[:8]],
              "first_board_orders": boards[:8]}
    path = folder / ("uncapped-replay.json" if args.uncapped else "capped-replay.json")
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "actions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
