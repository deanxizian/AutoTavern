"""Freeze, run and verify the simulator gates in docs/MODEL_READINESS.md.

This command never certifies a client adapter or human playing strength. Final
audits use new random seeds after copying and hashing the three trusted models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import secrets
import shutil
import statistics
import subprocess
import sys

from . import audit_bundles


SUITES = ("bots", "champion", "mixed")
BOT_DRIVER = "single_seat_batch_v1"
ARENA_DRIVER = "round_robin_player_views_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash():
    return hashlib.sha256(b"".join(
        p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
    )).hexdigest()


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")


def known_seeds(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if (key == "seed" or key.endswith("_seed")) and type(item) is int:
                found.add(item)
            elif (key == "seeds" or key.endswith("_seeds")) and isinstance(item, list):
                found.update(x for x in item if type(x) is int)
            found.update(known_seeds(item))
    elif isinstance(value, list):
        for item in value:
            found.update(known_seeds(item))
    return found


def draw_seeds(used):
    seeds = []
    while len(seeds) < 3:
        seed = secrets.randbelow(2**63 - 16) + 8
        if all(abs(seed - old) >= 8 for old in (*used, *seeds)):
            seeds.append(seed)
    return seeds


def freeze_seed_history(root, history_root, extra_paths, lineage_seeds, evidence_seeds, *, required):
    history_root = Path(history_root).resolve()
    require(not required or history_root.is_dir(), "Position audits require an existing seed history directory")
    paths = set(history_root.rglob("*.json")) | set(history_root.rglob("*.jsonl"))
    paths = {p.resolve() for p in paths if not p.resolve().is_relative_to(root)}
    paths.update(Path(p).resolve() for p in extra_paths)
    records, used = [], set(lineage_seeds) | set(evidence_seeds)
    for path in sorted(paths):
        digest, found = hashlib.sha256(), set()
        with path.open("rb") as handle:
            if path.suffix == ".jsonl":
                for raw in handle:
                    digest.update(raw)
                    if raw.strip():
                        found.update(known_seeds(json.loads(raw)))
            else:
                raw = handle.read()
                digest.update(raw)
                found.update(known_seeds(json.loads(raw)))
        records.append(dict(source_path=str(path), sha256=digest.hexdigest(), seeds=sorted(found)))
        used.update(found)
    index = dict(format_version=1, sources=records, model_lineage_seeds=sorted(lineage_seeds),
                 reference_evidence_seeds=sorted(evidence_seeds), excluded_seeds=sorted(used))
    target = root / "seed-history.json"
    write_new(target, index)
    return used, dict(path=target.name, sha256=checksum(target))


def verify_seed_history(root, manifest):
    record = manifest.get("seed_history")
    if record is None:
        require(manifest["format_version"] == 1, "Position audit omits seed history")
        return
    path = audit_bundles.inside(root, record["path"])
    require(checksum(path) == record["sha256"], "Frozen seed history changed")
    index = json.loads(path.read_text())
    require(index["format_version"] == 1, "Unsupported seed history index")
    used = set(index["model_lineage_seeds"]) | set(index["reference_evidence_seeds"])
    for source in index["sources"]:
        used.update(source["seeds"])
    require(sorted(used) == index["excluded_seeds"] == manifest["excluded_seeds"],
            "Frozen seed exclusion history differs")
    require(known_seeds({k: manifest[k] for k in audit_bundles.POLICY_KEYS}) <= used,
            "Seed history omits a model's lineage")


def validate_manifest(m):
    require(m.get("format_version") in (1, 2) and m.get("purpose") == "final_simulator_audit",
            "Unsupported audit manifest")
    require(len(m["seeds"]) == 3 and all(type(s) is int and 0 <= s < 2**63 for s in m["seeds"]),
            "Three valid audit seeds are required")
    require(all(abs(a - b) >= 8 for i, a in enumerate(m["seeds"]) for b in m["seeds"][:i]),
            "Audit seed ranges overlap")
    require(all(abs(a - b) >= 8 for a in m["seeds"] for b in m["excluded_seeds"]),
            "An audit seed overlaps recorded development or training seeds")
    identities = [m[k]["sha256"] for k in ("candidate", "champion", "reference")]
    require(len(set(identities)) == 3, "Candidate and the two references must be distinct")
    require(m["rules"].get("turn_action_caps") == [128], "Readiness requires the 128-action cap")
    if m["format_version"] == 1:
        require(m["decoding"] == audit_bundles.GREEDY,
                "This protocol freezes the validated greedy cancellation decoder v1")
        require(not any(m[k].get("position_bundle") for k in audit_bundles.POLICY_KEYS),
                "Position bundles require audit format 2")
    else:
        require(bool(m.get("seed_history")), "Position audit omits seed history")
        audit_bundles.validate_roles(m, known_seeds)
    for model in (m["candidate"], m["champion"], m["reference"]):
        require(model["engine_hash"] == m["engine_hash"] and model["action_cap"] == 128,
                "Model engine or action cap differs from the audit")
        require(all(abs(a - b) >= 8 for a in m["seeds"] for b in known_seeds(model)),
                "An audit seed overlaps a model's training lineage")
    require(bool(m["reference_evidence"]) and bool(m["reference_selection_note"].strip()),
            "Record evidence and the reason for choosing the strongest prior reference")
    require(datetime.fromisoformat(m["created_at"]).tzinfo is not None, "Manifest time lacks timezone")


def jobs(m):
    for suite in SUITES:
        for index, seed in enumerate(m["seeds"]):
            yield suite, seed, 512 if suite == "bots" else 256, f"{suite}-{index + 1}.json"


def role_ids(m, suite):
    champion, reference = m["champion"]["sha256"], m["reference"]["sha256"]
    opponents = ([champion] * 7 if suite == "champion" else
                 [champion] * 2 + [reference] * 2 + ["bot"] * 3)
    return [m["candidate"]["sha256"], *opponents]


def wilson(k, n):
    z = 1.959963984540054
    p, scale = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / scale
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return [max(0., center - half), min(1., center + half)]


def score(places):
    n, mean = len(places), statistics.mean(places)
    half = 1.96 * statistics.stdev(places) / math.sqrt(n)
    top4, wins = sum(p <= 4 for p in places), places.count(1)
    return dict(games=n, mean_placement=mean, placement_ci95=[mean - half, mean + half],
                top4_rate=top4 / n, top4_ci95=wilson(top4, n),
                win_rate=wins / n, win_ci95=wilson(wins, n))


def read_places(m, suite, seed, count, r):
    require(r.get("passed") is not False, "An assessment failed")
    require(r["seed"] == seed and r["games"] == count, "Assessment seed or game count differs")
    require(r["rules"] == m["rules"], "Assessment rules differ")
    require(r.get("guards_checked") is True, "Assessment did not check simulator guards")
    for field in ("truncated_combats", "turn_limit_ends", "rejected_actions", "incomplete_lobbies"):
        require(type(r.get(field)) is int and r[field] == 0, f"Assessment has missing/nonzero {field}")
    if m["format_version"] == 2:
        audit_bundles.verify_decoding(m, suite, r)
    else:
        require(all(r["decoding"].get(k) == v for k, v in m["decoding"].items()),
                "Assessment decoding differs")
    started = datetime.fromisoformat(r["evaluation_started_at"])
    require(started.tzinfo is not None and started >= datetime.fromisoformat(m["created_at"]),
            "Assessment predates the model and seed freeze")
    require(r["evaluation_code_hash"] == m["evaluation_code_hash"], "Assessment source differs")
    require(r["engine"]["source_hash"] == m["engine_hash"], "Assessment engine differs")
    if suite == "bots":
        require(r["driver"] == BOT_DRIVER and r["kind"] == "policy", "Wrong bot evaluation driver")
        require(r["checkpoint"]["sha256"] == m["candidate"]["sha256"], "Wrong bot candidate")
        raw = r["placements_by_seat"]
        require(set(raw) == {str(s) for s in range(8)}, "Bot evaluation omits a seat")
        require(all(len(raw[str(s)]) == count // 8 for s in range(8)), "Bot seats are unbalanced")
        places = [p for s in range(8) for p in raw[str(s)]]
        reported = r
    else:
        require(r["driver"] == ARENA_DRIVER, "Wrong learned-opponent driver")
        actual_ids = [x.get("sha256", "bot" if x.get("kind") == "bundled_heuristic" else None)
                      for x in r["role_models"]]
        require(actual_ids == role_ids(m, suite), "Wrong candidate or opponent roles")
        rows = r["lobbies"]
        require(len(rows) == count, "Arena omits lobbies")
        places = []
        for lane, row in enumerate(rows):
            require(row["lane"] == lane, "Arena repeats or reorders lanes")
            require(row["role_by_seat"] == [(seat - lane) % 8 for seat in range(8)],
                    "Arena roles are not balanced across seats")
            outcomes = row["placement_by_seat"]
            require(all(type(p) is int for p in outcomes) and sorted(outcomes) == list(range(1, 9)),
                    "Arena has an incomplete or invalid lobby")
            places.append(outcomes[lane % 8])
        reported = r["candidate"]
    require(all(type(p) is int and 1 <= p <= 8 for p in places), "Invalid placements")
    computed = score(places)
    for key in ("games", "mean_placement", "top4_rate", "win_rate"):
        require(math.isclose(reported[key], computed[key], rel_tol=0., abs_tol=1e-12),
                f"Reported {key} differs from raw lobbies")
    return places


def verify_reports(m, reports):
    validate_manifest(m)
    expected = {name for _, _, _, name in jobs(m)}
    require(set(reports) == expected, "The audit requires all nine assessments exactly once")
    results = {}
    for suite in SUITES:
        all_places, per_seed = [], []
        for kind, seed, count, name in jobs(m):
            if kind == suite:
                places = read_places(m, suite, seed, count, reports[name])
                all_places.extend(places)
                per_seed.append({"seed": seed, **score(places)})
        aggregate = score(all_places)
        upper, lower = aggregate["placement_ci95"][1], aggregate["top4_ci95"][0]
        checks = (dict(mean_upper_at_most_2_6=upper <= 2.6, top4_lower_at_least_80=lower >= .8,
                       win_lower_at_least_35=aggregate["win_ci95"][0] >= .35,
                       every_seed_mean_below_3=all(x["mean_placement"] < 3 for x in per_seed))
                  if suite == "bots" else
                  dict(mean_upper_below_4=upper < 4., top4_lower_at_least_60=lower >= .6,
                       every_seed_mean_below_4_5=all(x["mean_placement"] < 4.5 for x in per_seed)))
        results[suite] = dict(aggregate=aggregate, per_seed=per_seed, checks=checks,
                              passed=all(checks.values()))
    return dict(performance_gates_passed=all(r["passed"] for r in results.values()), live_ready=False,
                scope="Simulator readiness only; client validation and deployment checks remain separate",
                candidate_sha256=m["candidate"]["sha256"], total_games=3072, suites=results)


def freeze(args):
    import torch
    import taverngym as tg
    from .arena import frozen_checkpoint

    torch.set_num_threads(2)
    require(tg.build_is_current() is True, "Build the native extension from the current source first")
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    models, decoders = {}, {}
    version = 2 if any(getattr(args, f"{k}_position_bundle", None) for k in audit_bundles.POLICY_KEYS) else 1
    for key in ("candidate", "champion", "reference"):
        copied = root / f"{key}.pt"
        with Path(getattr(args, key)).open("rb") as source, copied.open("xb") as target:
            shutil.copyfileobj(source, target, 1024 * 1024)
        model, metadata = frozen_checkpoint(copied, torch.device("cpu"))
        models[key] = {**metadata, "path": copied.name}
        bundle = getattr(args, f"{key}_position_bundle", None)
        if bundle:
            models[key]["position_bundle"], decoders[key] = audit_bundles.freeze_bundle(
                bundle, root, key, model, metadata)
        else:
            decoders[key] = dict(audit_bundles.GREEDY)
        del model
    rules = {"turn_action_caps": [128]}
    if models["candidate"]["trained_hotfix_37_0_1"]:
        rules["hotfix_37_0_1"] = True
    evidence, seen_refs, evidence_seeds = [], set(), set()
    for index, path in enumerate(args.reference_evidence):
        copied = root / f"reference-evidence-{index + 1}.json"
        with path.open("rb") as source, copied.open("xb") as target:
            shutil.copyfileobj(source, target)
        data = json.loads(copied.read_text())
        require(data.get("guards_checked") is True and data.get("rules") == rules,
                "Reference selection evidence must use the audited rules and guards")
        require(data.get("driver") == ARENA_DRIVER, "Reference evidence uses another driver")
        if version == 2:
            require(data.get("passed") is not False and data["engine"]["source_hash"] == tg.source_hash(),
                    "Reference evidence failed or uses another engine")
            for field in ("truncated_combats", "turn_limit_ends", "rejected_actions", "incomplete_lobbies"):
                require(type(data.get(field)) is int and data[field] == 0, "Reference evidence has missing/nonzero guards")
            seen_refs.update(audit_bundles.evidence_identities(data))
        else:
            seen_refs.update(x.get("sha256") for x in data["role_models"])
        evidence_seeds.update(known_seeds(data))
        evidence.append(dict(path=copied.name, source_path=str(path.resolve()), sha256=checksum(copied)))
    references = {audit_bundles.policy_identity(models[k]["sha256"], decoders[k]) if version == 2
                  else models[k]["sha256"] for k in ("champion", "reference")}
    require(references <= seen_refs, "Reference evidence does not cover both chosen models and decoders")
    used, history = freeze_seed_history(root, args.history_root, getattr(args, "seed_history", []),
                                       known_seeds(models), evidence_seeds, required=version == 2)
    manifest = dict(format_version=version, purpose="final_simulator_audit", **models,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    engine_hash=tg.source_hash(), evaluation_code_hash=source_hash(), rules=rules,
                    decoding=dict(audit_bundles.ROLE_DECODING if version == 2 else audit_bundles.GREEDY),
                    excluded_seeds=sorted(used), seed_history=history, reference_evidence=evidence,
                    reference_selection_note=args.reference_selection_note)
    if version == 2:
        manifest["role_decoding"] = decoders
    manifest["seeds"] = draw_seeds(used)
    validate_manifest(manifest)
    write_new(root / "manifest.json", manifest)
    print(json.dumps(dict(manifest=str(root / "manifest.json"), seeds=manifest["seeds"])))


def load_audit(path):
    m = json.loads(path.read_text())
    validate_manifest(m)
    require(source_hash() == m["evaluation_code_hash"], "Audit code changed after the freeze")
    for key in ("candidate", "champion", "reference"):
        model_path = (audit_bundles.inside(path.parent, m[key]["path"]) if m["format_version"] == 2
                      else path.parent / m[key]["path"])
        require(checksum(model_path) == m[key]["sha256"], f"Frozen {key} changed")
        audit_bundles.verify_artifacts(path.parent, m[key])
    for evidence in m["reference_evidence"]:
        evidence_path = (audit_bundles.inside(path.parent, evidence["path"]) if m["format_version"] == 2
                         else path.parent / evidence["path"])
        require(checksum(evidence_path) == evidence["sha256"], "Reference evidence changed")
    verify_seed_history(path.parent, m)
    return m


def run_audit(path, *, device, threads):
    m, root = load_audit(path), path.parent
    require(not any((root / name).exists() or (root / name).with_suffix(".failed.json").exists()
                    for _, _, _, name in jobs(m)), "Preserve prior assessments; freeze a new audit")
    write_new(root / "execution.json", dict(started_at=datetime.now(timezone.utc).isoformat(),
                                           manifest_sha256=checksum(path)))
    try:
        for suite, seed, count, name in jobs(m):
            load_audit(path)
            command = [sys.executable, "-m", "training.evaluate" if suite == "bots" else "training.arena",
                       str(root / m["candidate"]["path"]), "--games", str(count), "--seed", str(seed),
                       "--threads", str(threads), "--device", device, "--cancel-guard",
                       "--hotfix-37-0-1" if m["rules"].get("hotfix_37_0_1") else "--no-hotfix-37-0-1",
                       "--out", str(root / name)]
            if m["candidate"].get("position_bundle"):
                command.extend(["--position-bundle", str(root / m["candidate"]["position_bundle"]["path"])])
            if suite != "bots":
                champion, reference = (str(root / m[k]["path"]) for k in ("champion", "reference"))
                roles = ([champion] * 7 if suite == "champion" else
                         [champion] * 2 + [reference] * 2 + ["bot"] * 3)
                role_keys = audit_bundles.suite_roles(suite)[1:]
                if any(k != "bot" and m[k].get("position_bundle") for k in role_keys):
                    bundles = [str(root / m[k]["position_bundle"]["path"])
                               if k != "bot" and m[k].get("position_bundle") else "none" for k in role_keys]
                    command.extend(["--opponent-position-bundles", *bundles])
                command.extend(["--opponents", *roles])
            with (root / name).with_suffix(".log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, check=True)
            print(json.dumps(dict(completed=name)), flush=True)
        verify_audit(path)
    except BaseException as exc:
        write_new(root / "execution.failed.json", dict(performance_gates_passed=False, live_ready=False, error=repr(exc)))
        raise


def verify_audit(path):
    m, root = load_audit(path), path.parent
    reports, evidence = {}, []
    for _, _, _, name in jobs(m):
        report = root / name
        require(not report.with_suffix(".failed.json").exists(), "An assessment has retained failure evidence")
        reports[name] = json.loads(report.read_text())
        evidence.append(dict(path=name, sha256=checksum(report)))
    result = verify_reports(m, reports)
    result.update(manifest_sha256=checksum(path), reports=evidence)
    output = root / "readiness.json"
    if output.exists():
        require(json.loads(output.read_text()) == result, "The retained readiness result differs")
    else:
        write_new(output, result)
    print(json.dumps(dict(performance_gates_passed=result["performance_gates_passed"], live_ready=False, total_games=3072)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze")
    p.add_argument("candidate", type=Path)
    p.add_argument("--champion", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--reference-evidence", type=Path, nargs="+", required=True)
    p.add_argument("--reference-selection-note", required=True)
    p.add_argument("--history-root", type=Path, default=Path("runs"))
    p.add_argument("--seed-history", type=Path, nargs="*", default=[],
                   help="Additional JSON/JSONL histories to exclude from final seed selection")
    for key in audit_bundles.POLICY_KEYS:
        p.add_argument(f"--{key}-position-bundle", type=Path)
    for name in ("run", "verify"):
        p = commands.add_parser(name)
        p.add_argument("manifest", type=Path)
        if name == "run":
            p.add_argument("--device", default="cuda")
            p.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
    elif args.command == "run":
        run_audit(args.manifest.resolve(), device=args.device, threads=args.threads)
    else:
        verify_audit(args.manifest.resolve())


if __name__ == "__main__":
    main()
