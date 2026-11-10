"""Publish compact V8+ stage evidence without copying journals or game records."""

import fcntl
import hashlib
import json
from pathlib import Path


def update_stage_overview(folder, config, *, completed=False):
    if config.get("metric_schema_version") != 2:
        return
    if config.get("lineage") == "eric-v7-3702-continuous":
        return update_v7_continuous_overview(folder, config, completed=completed)
    if config.get("lineage") == "eric-3702-fresh-v1":
        return update_fresh_overview(folder, config, completed=completed)
    from .experiment import now, write_json
    from .stage_chain import inspect_checkpoint

    root = Path(__file__).resolve().parents[3]
    folder = Path(folder).resolve()
    version = config["model_version"]
    lock = Path("/tmp") / ("autotavern-overview-" + hashlib.sha256(str(root).encode()).hexdigest()[:12])
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        path = root / "experiments/experiment-overview.json"
        overview = json.loads(path.read_text())
        progress_path = folder / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        values = {str(int(p.stem.rsplit("-", 1)[1])): json.loads(p.read_text())["mean_placement"]
                  for p in sorted(folder.glob("evaluation-[0-9][0-9][0-9][0-9].json"))}
        best_path = root / "runs" / config["name"] / config["best_model_filename"]
        best = inspect_checkpoint(best_path) if best_path.exists() else None
        row = {"experiment_dir": str(folder.relative_to(root)),
               "status": "completed" if completed else "running",
               "instance_id": config["instance_id"],
               "completed_ppo": progress.get("completed_iterations", 0), "planned_ppo": config["iterations"],
               "fresh_samples": progress.get("sampled_decisions", 0),
               "actual_new_adam_updates": progress.get("optimizer_steps", 0),
               "source_version": config["initial_model_version"],
               "source_checkpoint_sha256": config["initial_model_sha256"],
               "best_ppo": best["ppo_iteration"] if best else None,
               "best_adam_total": best["adam_total_steps"] if best else None,
               "checkpoint": str(best_path.relative_to(root)) if best else None,
               "checkpoint_sha256": best["sha256"] if best else None,
               "validation_placements": values,
               "roster": " + ".join(f"{s['count']} {s['model_version'].upper()}" for s in config["opponent_pool"]),
               "precision": "BF16", "validation_games_per_mode": config["evaluation_games"],
               "validation_modes": ["stochastic", "greedy"],
               "heldout_used_for_selection": False, "metric_schema_version": 2,
               "metric_schema_sha256": config["metric_schema_sha256"]}
        if completed:
            result = json.loads((folder / "results.json").read_text())
            row["results_sha256"] = hashlib.sha256((folder / "results.json").read_bytes()).hexdigest()
            row["heldout"] = {name: {condition: {key: value for key, value in item.items()
                if isinstance(value, (int, float, bool))} for condition, item in entry["conditions"].items()}
                for name, entry in result["heldout"].items()}
            lifecycle = json.loads((root / config["model_version_registry"]).read_text())["versions"][version]
            row.update(lifecycle_status=lifecycle["status"],
                       canonical_version=lifecycle["canonical_version"])
        overview.setdefault("stages", {})[version] = row
        overview["updated_at"] = now()
        overview["scope"] = overview.get("scope", "").replace("V8/V9 remain paused.", "V8/V9 new authorized finite instance recorded below; old drafts preserved.")
        write_json(path, overview)
        markdown = root / "experiments/EXPERIMENT_OVERVIEW.md"
        body = markdown.read_text().replace("V8/V9保持暂停。", "V8/V9新授权实例进度见文末；旧暂停草案保留。")
        start, end = "<!-- v89-live-summary:start -->", "<!-- v89-live-summary:end -->"
        block = start + "\n\n## V8→V9 有限授权实例（2026-11-03）\n\n"
        for key in ("v8", "v9"):
            stage = overview["stages"].get(key)
            if stage and stage.get("instance_id") == config["instance_id"]:
                block += (f"**{key.upper()}**：{stage['completed_ppo']}/100 PPO，新增Adam {stage['actual_new_adam_updates']}；"
                    f"对手 {stage['roster']}（0 V0），冻结随机；学习者随机/贪心各64局/val点，BF16。"
                    f"best PPO{stage['best_ppo']} / 累计Adam{stage['best_adam_total']}。"
                    f"本阶段随机val：{stage['validation_placements']}。训练与收尾 {stage['status']}；"
                    f"模型生命周期 {stage.get('lifecycle_status', 'unknown')}。"
                    f"[配置]({stage['experiment_dir'].removeprefix('experiments/')}/config.json)。\n\n")
        block += "各阶段val只在本阶段固定阵容内比较；独立test不选模。V9完成即停止，不进入V10。\n\n" + end
        if start in body:
            before, rest = body.split(start, 1)
            _, after = rest.split(end, 1)
            body = before + block + after
        else:
            body += "\n\n" + block + "\n"
        temp = markdown.with_suffix(".md.tmp")
        temp.write_text(body)
        temp.replace(markdown)


def update_fresh_overview(folder, config, *, completed=False):
    """Keep the fresh 37.0.2 lineage separate from the inherited V1-V9 history."""
    from .experiment import now, write_json
    from stable_baselines3.common.save_util import load_from_zip_file

    root = Path(__file__).resolve().parents[3]
    folder = Path(folder).resolve()
    run = root / "runs" / config["name"]
    best_path = run / config["best_model_filename"]
    best = None
    if best_path.exists():
        data, params, _ = load_from_zip_file(best_path, device="cpu")
        optimizer = params["policy.optimizer"]
        # The untrained start is eligible for best and correctly has no Adam state.
        clocks = [int(state["step"]) for state in optimizer["state"].values()]
        best = {"path": str(best_path.relative_to(root)),
                "sha256": hashlib.sha256(best_path.read_bytes()).hexdigest(),
                "ppo": data["num_timesteps"] // config["decisions_per_iteration"],
                "samples": data["num_timesteps"], "adam": max(clocks, default=0)}
    progress_path = folder / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    training_path = folder / "training.json"
    if training_path.exists():
        progress = json.loads(training_path.read_text())
    values = {str(int(p.stem.rsplit("-", 1)[1])): json.loads(p.read_text())["mean_placement"]
              for p in sorted(folder.glob("evaluation-[0-9][0-9][0-9][0-9].json"))}
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(folder.glob("evaluation-*.json"))}
    boundary_ppo = 0
    latest = run / "latest_model.zip"
    if latest.exists():
        latest_data, _, _ = load_from_zip_file(latest, device="cpu")
        budget = config["decisions_per_iteration"]
        if (latest_data["num_timesteps"] % budget
                or latest_data["_n_updates"] != latest_data["num_timesteps"] // budget * config["n_epochs"]):
            raise ValueError("Latest checkpoint is not a complete PPO boundary")
        boundary_ppo = latest_data["num_timesteps"] // budget
    row = {"experiment_dir": str(folder.relative_to(root)),
           "status": "completed" if completed else progress.get("status", "running"),
           "instance_id": config["instance_id"], "rules_version": config["rules_version"],
           "source": "random initialization; no inherited historical policy or Adam",
           "initial_model_sha256": config["initial_model_sha256"],
           "completed_ppo": boundary_ppo,
           "completed_boundary_samples": boundary_ppo * config["decisions_per_iteration"],
           "planned_ppo": config["iterations"],
           "fresh_samples": progress.get("sampled_decisions", 0),
           "actual_new_adam_updates": progress.get("optimizer_steps", 0),
           "best": best, "validation_placements": values, "roster": "7 V0",
           "precision": "FP32 parameters / BF16 forward",
           "validation_games_per_mode": config["evaluation_games"],
           "validation_modes": ["stochastic", "greedy"],
           "heldout_used_for_selection": False,
           "metric_schema_version": 2, "metric_schema_sha256": config["metric_schema_sha256"],
           "source_hashes": sources}
    heldout_path = folder / "evaluation-best-heldout.json"
    if heldout_path.exists():
        result = json.loads(heldout_path.read_text())
        row["heldout"] = {name: {key: value for key, value in entry.items()
                                if isinstance(value, (int, float, bool))}
                          for name, entry in result["conditions"].items()}
    lock = Path("/tmp") / ("autotavern-overview-" + hashlib.sha256(str(root).encode()).hexdigest()[:12])
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        path = root / "experiments/experiment-overview.json"
        overview = json.loads(path.read_text())
        overview.setdefault("fresh_lineages", {})[config["lineage"]] = row
        overview["updated_at"] = now()
        write_json(path, overview)
        markdown = root / "experiments/EXPERIMENT_OVERVIEW.md"
        body = markdown.read_text()
        start, end = "<!-- eric-3702-fresh:start -->", "<!-- eric-3702-fresh:end -->"
        selected = f"PPO{best['ppo']} / Adam{best['adam']}" if best else "尚未选择"
        block = (start + "\n\n## Eric 37.0.2 全新谱系（2026-11-08）\n\n"
                 + f"eric-3702-fresh-v1：{row['status']}，{row['completed_ppo']}/100 PPO，"
                 + f"{row['fresh_samples']} 新样本，Adam {row['actual_new_adam_updates']}。"
                 + f"随机初始化，未继承旧模型；对手7 V0，规则37.0.2，英雄二选一，BF16前向。"
                 + f"随机/贪心各64局/开发点；best {selected}；开发随机均名 {values}。"
                 + "结束best独立随机/贪心各128局，不参与选模。"
                 + f"[配置]({folder.relative_to(root / 'experiments')}/config.json)。"
                 + "本谱系不与旧补丁或不同阵容的历史指标横比。\n\n" + end)
        if start in body:
            before, rest = body.split(start, 1)
            _, after = rest.split(end, 1)
            body = before + block + after
        else:
            body += "\n\n" + block + "\n"
        temporary = markdown.with_suffix(".md.tmp")
        temporary.write_text(body)
        temporary.replace(markdown)


def update_v7_continuous_overview(folder, config, *, completed=False):
    """Record the V7-derived goal separately from the stopped fresh and historical V7."""
    from stable_baselines3.common.save_util import load_from_zip_file
    from .experiment import now, write_json

    root = Path(__file__).resolve().parents[3]
    folder = Path(folder).resolve()
    run = root / "runs" / config["name"]

    def checkpoint(path):
        if not path.exists():
            return None
        data, params, _ = load_from_zip_file(path, device="cpu")
        steps = [int(row["step"]) for row in params["policy.optimizer"]["state"].values()]
        return {"path": str(path.relative_to(root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "ppo": data["num_timesteps"] // config["decisions_per_iteration"],
                "stage_samples": data["num_timesteps"],
                "stage_adam": max(steps, default=0) - data.get("_optimizer_step_offset", 0),
                "cumulative_adam": max(steps, default=0)}

    progress_path = folder / ("training.json" if completed else "progress.json")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    latest = checkpoint(run / "latest_model.zip")
    selected = checkpoint(run / config["best_model_filename"])
    values = {str(int(path.stem.rsplit("-", 1)[1])):
              json.loads(path.read_text())["mean_placement"] for path in
              sorted(folder.glob("evaluation-[0-9][0-9][0-9][0-9].json"))}
    roster = " + ".join(f"{item['count']} {item['canonical_version'].upper()}"
                        for item in config["opponent_pool"]) + " + 1 V0"
    row = {"experiment_dir": str(folder.relative_to(root)),
           "instance_id": config["instance_id"], "rules": config["rules_version"],
           "status": "stage_completed_not_goal_complete" if completed
                     else progress.get("status", "running"),
           "initial_source": config["initial_model_version"],
           "initial_source_sha256": config["initial_model_sha256"],
           "source_cumulative_adam": config["source_adam_total_steps"],
           "completed_ppo": latest["ppo"] if latest else 0,
           "planned_ppo": config["iterations"],
           "actual_stage_sampled_decisions": progress.get("sampled_decisions", 0),
           "actual_stage_adam": progress.get("optimizer_steps", 0),
           "latest_complete": latest, "selected_stage_best": selected,
           "validation_placements": values, "opponents": roster,
           "validation_modes": ["sampled", "greedy"],
           "validation_games_each_mode": config["evaluation_games"],
           "independent_tests_choose_champion": False,
           "evaluation_protocol_sha256": config["evaluation_protocol_sha256"]}
    lock = Path("/tmp") / ("autotavern-overview-" + hashlib.sha256(str(root).encode()).hexdigest()[:12])
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        path = root / "experiments/experiment-overview.json"
        overview = json.loads(path.read_text())
        goal = overview.setdefault("continuous_training_goals", {}).setdefault(
            config["lineage"], {"initial_source": "adapted Eric V7; full Adam8800"})
        goal.setdefault("stages", {})[config["model_version"]] = row
        goal["native_goal_status"] = "active; stage completion does not complete Goal"
        overview["updated_at"] = now()
        write_json(path, overview)
        markdown = root / "experiments/EXPERIMENT_OVERVIEW.md"
        text = markdown.read_text()
        start, end = "<!-- eric-v7-continuous:start -->", "<!-- eric-v7-continuous:end -->"
        block = start + "\n\n## Eric V7 37.0.2 持续Goal\n\n"
        for version, item in goal["stages"].items():
            block += (f"{version}：{item['status']}，完整PPO{item['completed_ppo']}/{item['planned_ppo']}，"
                      f"阶段样本{item['actual_stage_sampled_decisions']}，新增Adam{item['actual_stage_adam']}；"
                      f"来源{item['initial_source']}完整Adam{item['source_cumulative_adam']}。"
                      f"对手{item['opponents']}，随机/贪心各{item['validation_games_each_mode']}局/val点，BF16。"
                      f"随机val：{item['validation_placements']}；阶段best并非已晋级冠军。\n\n")
        block += "强池多种子开发晋级与独立交付test分离；仅阶段结束不代表Goal完成。\n\n" + end
        if start in text:
            before, rest = text.split(start, 1)
            _, after = rest.split(end, 1)
            text = before + block + after
        else:
            text += "\n\n" + block + "\n"
        temporary = markdown.with_suffix(".md.tmp")
        temporary.write_text(text)
        temporary.replace(markdown)
