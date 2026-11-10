"""Benchmark the frozen V1 and trained V2 PPO; seal answers before scoring."""

import collections
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

SOURCE = Path(__file__).resolve().parent
ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT / "tavernbench"))

import numpy as np
import torch
import taverngym
from taverngym.learning.ppo import EntityPPO

from adapter import COMBAT, RULES, predict


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def run():
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    out = SOURCE if not (SOURCE / "inference.jsonl").exists() else SOURCE / "reruns" / stamp
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / "tavernbench/data/eval"
    model_paths = {
        "ppo-v1": ROOT / "runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip",
        "ppo-v2": ROOT / "runs/20261031-ppo-mixed-v2" / json.loads((SOURCE.parent / "20261031-ppo-mixed-v2/resolved-config.json").read_text())["best_model_filename"],
    }
    hashes = {name: digest(path) for name, path in model_paths.items()}
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    started = time.perf_counter()
    metadata = {"id": SOURCE.name, "date": "2026-10-31", "timezone": "Asia/Shanghai",
                "status": "predicting", "protocol": "protocol.json", "models": {
                    name: {"path": str(path.relative_to(ROOT)), "sha256": hashes[name]}
                    for name, path in model_paths.items()}, "weights_uploaded": False}
    write(out / "experiment.json", metadata)

    # Deny label/reference-answer reads until every prediction file has been sealed.
    predicting = [True]

    def deny_labels(event, args):
        if predicting[0] and event == "open" and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if "labels.jsonl" in path or "tavernbench/results/" in path:
                raise RuntimeError("Inference attempted to open labels or reference answers")

    sys.addaudithook(deny_labels)
    prompts = [json.loads(line) for line in (data / "prompts.jsonl").read_text().splitlines()]
    with gzip.open(data / "states.jsonl.gz", "rt") as stream:
        specs = {row["id"]: row for row in map(json.loads, stream)}
    assert len(prompts) == 368 and len({row["id"] for row in prompts}) == 368
    allowed = {"id", "state", "states", "options", "perms", "answers", "map", "seat", "health", "cap"}
    for spec in specs.values():
        assert set(spec) <= allowed
        for option in spec.get("options", []):
            assert set(option) == {"answer", "ops"}
    seeds = {row["id"]: int.from_bytes(hashlib.blake2b(
        ("scripted-v1:" + row["id"]).encode(), digest_size=8).digest(), "little") for row in prompts}
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert taverngym.build_is_current() is True
    inference = {}
    with (out / "inference.jsonl").open("x", buffering=1) as stream:
        for name, path in model_paths.items():
            # Inference does not need the checkpoint's 256 x 1024 rollout buffer.
            model = EntityPPO.load(path, device=device, n_envs=1, n_steps=2, batch_size=2)
            model.policy.set_training_mode(False)
            inference[name] = []
            for i, prompt in enumerate(prompts, 1):
                begin = time.perf_counter()
                try:
                    answer = predict(model, prompt["scenario"], specs.get(prompt["id"], {}), seeds[prompt["id"]])
                except Exception as exc:
                    answer = {"status": "adapter_error", "response": None,
                              "reason": f"{type(exc).__name__}: {exc}"}
                answer.update(id=prompt["id"], scenario=prompt["scenario"], model=name,
                              latency_s=time.perf_counter() - begin)
                stream.write(json.dumps(answer, ensure_ascii=False) + "\n")
                inference[name].append(answer)
                if i % 40 == 0 or i == len(prompts):
                    print(name, i, "/", len(prompts), dict(collections.Counter(
                        row["status"] for row in inference[name])), flush=True)
            close = getattr(model.rollout_buffer, "close", None)
            if close is not None:
                close()
            del model
    errors = [row for rows in inference.values() for row in rows if row["status"] == "adapter_error"]
    if errors:
        write(out / "adapter-errors.json", errors)
        raise RuntimeError(f"{len(errors)} adapter errors; labels have not been opened")

    binary = ROOT / "target/scripted-tavernbench"
    tests = subprocess.run([str(ROOT / "target/scripted-tavernbench-tests"), "--test-threads=1"],
                           capture_output=True, text=True, check=True)
    (out / "combat-adapter-tests.txt").write_text(tests.stdout)
    requests = [{"id": row["id"], "scenario": row["scenario"], "seed": seeds[row["id"]],
                 "spec": specs[row["id"]]} for row in prompts if row["scenario"] in COMBAT]
    generated = subprocess.run([str(binary)], input="".join(json.dumps(row) + "\n" for row in requests),
                               capture_output=True, text=True, check=True)
    (out / "combat-inference.jsonl").write_text(generated.stdout)
    combat_rows = [json.loads(line) for line in generated.stdout.splitlines()]
    assert len(combat_rows) == 56 and not any(row.get("panic") for row in combat_rows)
    combat = {row["id"]: row["combat256"] for row in combat_rows}
    assert all(row["status"] == "ok" for row in combat.values())
    files = {}
    for name, predictions in inference.items():
        for suffix in ("policy", "combat256"):
            system, answers = name + "-" + suffix, []
            for original in predictions:
                prediction = combat[original["id"]] if suffix == "combat256" and original["scenario"] in COMBAT else original
                ok = prediction["status"] == "ok"
                answers.append({"id": original["id"], "model": system,
                                "response": json.dumps(prediction["response"]) if ok else None,
                                "error": None if ok else prediction.get("reason", prediction["status"]),
                                "status": prediction["status"], "finish_reason": "stop" if ok else None,
                                "latency_s": prediction["latency_s"], "input_tokens": 0, "output_tokens": 0})
            folder = out / "answers" / system
            folder.mkdir(parents=True)
            path = folder / "answers.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in answers))
            files[system] = path
    manifest = {"sealed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "protocol": protocol, "device": device, "native_build": taverngym.build_info(),
                "input_hashes": {name: digest(data / name) for name in ("prompts.jsonl", "states.jsonl.gz")},
                "model_hashes": hashes,
                "source_hashes": {str(path.relative_to(ROOT)): digest(path) for path in
                                  (SOURCE / "run.py", SOURCE / "adapter.py", SOURCE / "protocol.json")},
                "combat_binary_sha256": digest(binary),
                "combat_source_sha256": digest(ROOT / "experiments/20261031-scripted-baseline/20261031-scripted-bot-tavernbench/adapter.rs"),
                "answers_sha256": {name: digest(path) for name, path in files.items()},
                "labels_read_by_inference": False, "candidate_score_values_used": False,
                "inference_wall_seconds": time.perf_counter() - started}
    write(out / "prediction-manifest.json", manifest)
    predicting[0] = False
    print("Predictions sealed; labels may now be opened for official scoring.", flush=True)

    from tavernbench.report import load, evaluate, summarize
    from tavernbench.score import load_answers, score_answer

    labels, prompt_map = load(data)
    summary, items = {}, []
    for system, path in files.items():
        acc, cost = evaluate(labels, prompt_map, path)
        answers = load_answers(path)
        coverage = collections.Counter()
        for task_id, prompt in prompt_map.items():
            row = answers[task_id]
            correct, graded = score_answer(labels[task_id], row)
            coverage[row["status"]] += 1
            if prompt["scenario"] not in RULES:
                coverage["play_total"] += 1
                coverage["play_ok"] += row["status"] == "ok"
                coverage["play_correct"] += correct
            items.append({"system": system, "id": task_id, "scenario": prompt["scenario"],
                          "status": row["status"], "correct": correct, "graded_score": graded})
        strategy = [value for sid, value in acc.items() if sid not in RULES | COMBAT]
        summary[system] = {**summarize(acc), **cost, "by_scenario": acc,
                           "coverage": dict(coverage), "rules_applicable": False,
                           "strategy_23_scenario_mean": sum(strategy) / len(strategy)}
        assert digest(path) == manifest["answers_sha256"][system]
        print(system, "Play", f"{summary[system]['play']:.4%}", dict(coverage), flush=True)
    references = {}
    for system in ("scripted-policy", "scripted-combat256"):
        path = ROOT / "experiments/20261031-scripted-baseline/20261031-scripted-bot-tavernbench/answers" / system / "answers.jsonl"
        acc, _ = evaluate(labels, prompt_map, path)
        references[system] = {**summarize(acc), "by_scenario": acc, "answers_sha256": digest(path)}
    result = {"dataset": "eval", "tasks": 368, "play_tasks": 288, "play_scenarios": 30,
              "systems": summary, "scripted_references": references,
              "labels_sha256": digest(data / "labels.jsonl"), "models_unchanged": True,
              "rules_interpretation": "not applicable; unsupported items are zero in the unchanged official denominator"}
    for name, path in model_paths.items():
        assert digest(path) == hashes[name]
    write(out / "results.json", result)
    (out / "item-scores.jsonl").write_text("".join(json.dumps(row) + "\n" for row in items))
    with (out / "per-scenario.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario", *summary])
        for sid in sorted(next(iter(summary.values()))["by_scenario"]):
            writer.writerow([sid, *[row["by_scenario"][sid] for row in summary.values()]])
    report = subprocess.run([sys.executable, "-m", "tavernbench.report", "--set", str(data),
                             "--runs", str(out / "answers"), "--systems", ",".join(files),
                             "--out", str(out / "official-report.json")],
                            cwd=ROOT / "tavernbench", capture_output=True, text=True, check=True)
    (out / "official-report.txt").write_text(report.stdout)
    write(out / "validation.json", {"status": "passed", "tasks_per_system": 368,
                                    "unique_ids": True, "prediction_hashes_unchanged": True,
                                    "model_hashes_unchanged": True, "adapter_errors": 0,
                                    "labels_denied_during_inference": True})
    metadata.update(status="completed", completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                    results="results.json", elapsed_seconds=time.perf_counter() - started)
    write(out / "experiment.json", metadata)
    print("Saved", out, flush=True)


if __name__ == "__main__":
    run()
