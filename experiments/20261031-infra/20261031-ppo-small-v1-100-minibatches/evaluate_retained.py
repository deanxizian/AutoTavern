"""Validate the preselected best policy on held-out seeds; never update weights."""

import json
from pathlib import Path

import torch

from taverngym.learning.experiment import evaluate
from taverngym.learning.ppo import EntityPPO


def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    folder = Path(__file__).resolve().parent
    models = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file()) / "runs" / folder.name
    config = json.loads((folder / "config.json").read_text())
    evaluations = [(int(path.stem.split("-")[-1]), json.loads(path.read_text()))
                   for path in sorted(folder.glob("evaluation-[0-9][0-9][0-9][0-9].json"))]
    selected_iteration, selected = min(evaluations, key=lambda item: item[1]["mean_placement"])
    results = {}
    for name, path in (("best", models / "best_model.zip"),
                       ("initial", models.parent / "20261031-ppo-small-init/initial_model.zip")):
        model = EntityPPO.load(path, device="cuda")
        if name == "best":
            assert model.num_timesteps == selected_iteration * 8192
        result = evaluate(model, config["final_evaluation_games"], config["final_evaluation_seed"],
                          simulator_config=config["simulator_config"])
        result.update(model=name, model_timesteps=model.num_timesteps,
                      model_training_updates=model.num_timesteps // 8192,
                      used_for_model_selection=False)
        (folder / f"evaluation-{name}-heldout.json").write_text(json.dumps(result, indent=2) + "\n")
        results[name] = result
        print(name, {key: result[key] for key in ("games", "mean_placement", "first_rate", "top4_rate")}, flush=True)
        del model
    results["selection"] = {"iteration": selected_iteration, "games": selected["games"],
                            "seed": selected["seed"], "mean_placement": selected["mean_placement"],
                            "method": "lowest fixed-seed 64-game mean placement; earliest wins ties; held-out results never select"}
    (folder / "heldout-comparison.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
