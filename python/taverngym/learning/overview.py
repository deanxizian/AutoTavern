"""Publish compact V8+ stage evidence without copying journals or game records."""

import fcntl
import hashlib
import json
from pathlib import Path


def update_stage_overview(folder, config, *, completed=False):
    if config.get("metric_schema_version") != 2:
        return
    from .experiment import now, write_json
    from .stage_chain import inspect_checkpoint

    root = Path(__file__).resolve().parents[3]
    folder = Path(folder)
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
        overview["stages"][version] = row
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
