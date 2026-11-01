"""Original V3, complete development games under the final soft-cycle protocol."""

from pathlib import Path
import math

import torch

from taverngym.learning.experiment import evaluate_suite, load_frozen_opponents, now, write_json
from taverngym.learning.ppo import EntityPPO, PlacementBuffer
from taverngym.learning.stage_chain import digest, read_json, validate_evaluation


def main():
    folder = Path(__file__).resolve().parent
    manifest = read_json(folder / "manifest.json")
    root = Path(manifest["root"])
    config = read_json(root / manifest["stages"][0]["experiment_dir"] / "config.json")
    source = root / config["initial_model"]
    source_hash = digest(source)
    assert source_hash == config["initial_model_sha256"]
    assert config["reward_shaping"] == {"operation_cost": 0, "cycle_cost": .01,
        "cycle_detection": "all_actions", "cycle_forfeit_repetitions": None,
        "turn_action_forfeit_limit": None}
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    model = EntityPPO.load(source, device="cuda", n_envs=1, n_steps=2, batch_size=2,
                          rollout_buffer_class=PlacementBuffer, rollout_buffer_kwargs={})
    source_adam = model.optimizer_total_step_count()
    opponents = load_frozen_opponents(config, "cuda")
    result = evaluate_suite(model, 64, config["evaluation_seed"], config, opponents)
    write_json(folder / "development-initial-raw.json", result)
    validate_evaluation(result, 64, config["evaluation_seed"], config["opponent_pool"])
    summary = {}
    for name, condition in result["conditions"].items():
        assert condition["cycle_forfeit_rate"] == condition["action_limit_forfeit_rate"] == 0
        records = condition["episodes"]
        assert len(records) == 64
        for episode in records:
            components = episode["reward_components"]
            assert episode["termination_reason"] == "native_terminal"
            assert episode["turn_before_terminal_action"] >= 1
            assert math.isclose(components["cycle_penalty"], -.01 * components["cycle_returns"],
                                abs_tol=1e-9)
            assert math.isclose(components["cycle_return_fraction"],
                components["cycle_returns"] / components["actions_per_game"], abs_tol=1e-12)
            assert components["actions_per_game"] == episode["episode"]["l"]
        summary[name] = {key: condition[key] for key in (
            "games", "mean_placement", "first_rate", "top4_rate", "cycle_returns",
            "cycle_return_fraction", "max_consecutive_cycle_returns", "cycle_penalty",
            "actions_per_game", "cap_end_rate", "cycle_forfeit_rate", "action_limit_forfeit_rate")}
        summary[name]["terminal_turn_range"] = [
            min(record["turn_before_terminal_action"] for record in records),
            max(record["turn_before_terminal_action"] for record in records)]
        summary[name]["natural_terminal_games"] = len(records)
        summary[name]["natural_first_turn_games"] = sum(
            record["turn_before_terminal_action"] == 1 for record in records)
    assert digest(source) == source_hash
    assert model.optimizer_total_step_count() == source_adam == 5600
    write_json(folder / "development-initial-validation.json", {"evaluated_at": now(),
        "purpose": "preflight development validation; not independent heldout and no selection",
        "source_model": str(source), "source_sha256": source_hash, "adam_unchanged": True,
        "config_sha256": digest(root / manifest["stages"][0]["experiment_dir"] / "config.json"),
        "opponents": config["opponent_pool"], "scripted_opponents": 1,
        "reward_shaping": config["reward_shaping"], "summary": summary, "evaluation": result})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
