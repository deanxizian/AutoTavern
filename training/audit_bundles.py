"""Frozen public-position components and per-role evidence for simulator audits.

Verification uses only the standard library. Torch is needed only while freezing
trusted bundles, when the actual loader validates their component bindings.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import shutil


POLICY_KEYS = ("candidate", "champion", "reference")
GREEDY = {"kind": "greedy", "cancel_retry_guard": True, "guard_version": 1}
POSITION_KIND = "greedy_plus_public_position_v1"
ROLE_DECODING = {"kind": "frozen_role_decoders_v1", "cancel_retry_guard": True, "guard_version": 1}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def inside(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Frozen audit paths must be relative and stay inside their directory")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Frozen audit path escapes its directory")
    return path


def normalize_decoder(value):
    # Display text does not change behavior. Every semantic field remains bound.
    return {key: item for key, item in value.items() if key != "scope"}


def expected_position_decoder(bundle):
    config = bundle["config"]
    return dict(GREEDY, kind=POSITION_KIND, position_version=1,
                actor_sha256=config["actor"]["sha256"],
                encoder_sha256=config["encoder"]["sha256"],
                ridge_sha256=config["ridge"]["sha256"], threshold=config["threshold"],
                encoder_precision="float32_highest", bundle_sha256=bundle["sha256"])


def validate_roles(manifest, known_seeds):
    require(manifest["decoding"] == ROLE_DECODING, "Unsupported frozen role decoding protocol")
    require(set(manifest["role_decoding"]) == set(POLICY_KEYS), "Freeze all three role decoders")
    require(any(manifest[key].get("position_bundle") for key in POLICY_KEYS),
            "The position audit needs at least one position bundle")
    require(set(manifest["rules"]) <= {"turn_action_caps", "hotfix_37_0_1"},
            "Unsupported audit rules")
    for key in POLICY_KEYS:
        model = manifest[key]
        bundle = model.get("position_bundle")
        wanted = GREEDY
        if bundle:
            config = bundle["config"]
            bundle_path = inside(Path("/audit"), bundle["path"])
            inside(Path("/audit"), model["path"])
            require(config.get("format_version") == 1 and config.get("kind") == "public_position_v1",
                    "Unsupported frozen position bundle")
            require(config["engine_hash"] == manifest["engine_hash"], "Position bundle engine differs")
            require(set(config["rules"]) <= {"turn_action_caps", "hotfix_37_0_1"}
                    and config["rules"].get("turn_action_caps") == [128]
                    and bool(config["rules"].get("hotfix_37_0_1")) == bool(manifest["rules"].get("hotfix_37_0_1")),
                    "Position bundle rules differ")
            require(config["actor"]["sha256"] == model["sha256"], "Position bundle actor differs")
            threshold = config["threshold"]
            require(type(threshold) in (int, float) and math.isfinite(threshold) and threshold >= 0,
                    "Invalid position threshold")
            encoder = bundle["encoder_metadata"]
            require(encoder["sha256"] == config["encoder"]["sha256"], "Position encoder identity differs")
            require(inside(Path("/audit"), encoder["path"]) == inside(bundle_path.parent, config["encoder"]["path"]),
                    "Position encoder metadata points to another component")
            for metadata in (model, encoder):
                require(metadata["engine_hash"] == manifest["engine_hash"] and metadata["action_cap"] == 128,
                        "Position model engine or action cap differs")
                require(bool(metadata.get("trained_hotfix_37_0_1")) == bool(config["rules"].get("hotfix_37_0_1")),
                        "Position model training profile differs from its bundle")
            require(all(abs(seed - old) >= 8 for seed in manifest["seeds"] for old in known_seeds(encoder)),
                    "An audit seed overlaps the position encoder's training lineage")
            for name in ("actor", "encoder", "ridge"):
                inside(Path("/audit-bundle"), config[name]["path"])
            wanted = expected_position_decoder(bundle)
        require(manifest["role_decoding"][key] == wanted, f"Frozen {key} decoder differs from its components")


def freeze_bundle(source, root, key, actor, metadata):
    """Copy only the manifest and its three validated components, retaining bytes."""
    import torch
    from .arena import frozen_checkpoint
    from .position import load_position_policy

    source, root = Path(source), Path(root)
    raw = source.read_bytes()
    config = json.loads(raw)
    wrapped = load_position_policy(actor, metadata, source, torch.device("cpu"))
    digest = hashlib.sha256(raw).hexdigest()
    require(wrapped.decoder_metadata["bundle_sha256"] == digest, "Position bundle changed during freezing")
    directory = root / f"{key}-position"
    directory.mkdir()
    target = directory / "position.json"
    with target.open("xb") as handle:
        handle.write(raw)
    for name in ("actor", "encoder", "ridge"):
        info = config[name]
        original, copied = inside(source.parent, info["path"]), inside(directory, info["path"])
        if not copied.exists():
            copied.parent.mkdir(parents=True, exist_ok=True)
            with original.open("rb") as first, copied.open("xb") as second:
                shutil.copyfileobj(first, second, 1 << 20)
        require(checksum(copied) == info["sha256"], f"Frozen position {name} changed while copying")
    # Validate the portable copy with the same loader used by the real evaluators.
    copied_policy = load_position_policy(actor, metadata, target, torch.device("cpu"))
    encoder_path = inside(directory, config["encoder"]["path"])
    _, encoder = frozen_checkpoint(encoder_path, torch.device("cpu"))
    encoder["path"] = str(encoder_path.relative_to(root.resolve()))
    bundle = dict(path=str(target.relative_to(root)), sha256=digest, config=config, encoder_metadata=encoder)
    decoder = normalize_decoder(dict(GREEDY, **copied_policy.decoder_metadata))
    require(decoder == expected_position_decoder(bundle), "Loaded position decoder differs from the frozen bundle")
    return bundle, decoder


def verify_artifacts(root, model):
    bundle = model.get("position_bundle")
    if not bundle:
        return
    manifest = inside(root, bundle["path"])
    require(checksum(manifest) == bundle["sha256"], "Frozen position manifest changed")
    require(json.loads(manifest.read_text()) == bundle["config"], "Frozen position configuration differs")
    for name in ("actor", "encoder", "ridge"):
        component = bundle["config"][name]
        require(checksum(inside(manifest.parent, component["path"])) == component["sha256"],
                f"Frozen position {name} changed")


def suite_roles(suite):
    return ["candidate", *(["champion"] * 7 if suite == "champion" else
                           ["champion"] * 2 + ["reference"] * 2 + ["bot"] * 3)]


def policy_identity(actor_sha, decoder):
    """Component identity for reference selection, independent of packaging paths."""
    require(decoder.get("cancel_retry_guard") is True and decoder.get("guard_version") == 1,
            "Reference evidence lacks the frozen cancellation guard")
    if decoder.get("kind") == "greedy":
        return (actor_sha, "greedy", 1)
    require(decoder.get("kind") in (POSITION_KIND, "public_position_v1"),
            "Unsupported decoder in reference selection evidence")
    require(decoder.get("actor_sha256") == actor_sha, "Reference decoder actor differs")
    require(decoder.get("position_version", 1) == 1 and decoder.get("encoder_precision") == "float32_highest",
            "Reference position decoder version or precision differs")
    return (actor_sha, POSITION_KIND, 1, decoder["encoder_sha256"], decoder["ridge_sha256"], decoder["threshold"])


def evidence_identities(report):
    models = report["role_models"]
    require(len(models) == 8, "Reference evidence must identify all eight roles")
    if "role_decoding" in report:
        decoders = report["role_decoding"]
    elif "role_policies" in report:
        decoders = report["role_policies"]
    else:
        common = report["decoding"]
        require(common.get("kind") in ("greedy", POSITION_KIND), "Unidentified reference decoders")
        decoders = [common, *[dict(GREEDY, cancel_retry_guard=common.get("cancel_retry_guard"),
                                   guard_version=common.get("guard_version")) for _ in range(7)]]
    require(len(decoders) == 8, "Reference evidence omits a role decoder")
    result = set()
    for model, decoder in zip(models, decoders):
        if model.get("kind") == "bundled_heuristic" or model.get("sha256") == "bot":
            continue
        result.add(policy_identity(model["sha256"], decoder))
    return result


def check_event(event, *, identities, threshold, seen):
    identity, turn, action = event["identity"], event["turn"], event["action"]
    require(type(identity) is int and 0 <= identity < identities, "Position event has an invalid slot")
    require(type(turn) is int and 4 <= turn <= 20, "Position event has an ineligible turn")
    require(type(action) is int and 196 <= action < 245, "Position event is not a Move")
    source, target = divmod(action - 196, 7)
    require(source != target, "Position event moves a minion to its own slot")
    gain = event["predicted_gain"]
    require(type(gain) in (int, float) and math.isfinite(gain) and gain > threshold,
            "Position event fails the frozen threshold")
    require((identity, turn) not in seen, "Repeated position Move in the same slot and turn")
    seen.add((identity, turn))


def verify_decoding(manifest, suite, report):
    wanted = manifest["role_decoding"]
    if suite == "bots":
        require(normalize_decoder(report["decoding"]) == wanted["candidate"], "Bot candidate decoder differs")
        position = wanted["candidate"]["kind"] == POSITION_KIND
        require(("position_decoder" in report) == position, "Bot position evidence differs")
        if position:
            events = report["position_decoder"]["events"]
            require(report["position_decoder"]["counted_episode_moves"] == len(events), "Bot Move count differs")
            seen = set()
            for event in events:
                check_event(event, identities=report["games"], threshold=wanted["candidate"]["threshold"], seen=seen)
        return
    roles = suite_roles(suite)
    expected = [wanted[key] if key != "bot" else {"kind": "bundled_heuristic"} for key in roles]
    require([normalize_decoder(d) for d in report["role_decoding"]] == expected, "Arena role decoders differ")
    positioned = {role for role, decoder in enumerate(expected) if decoder["kind"] == POSITION_KIND}
    header = report["decoding"]
    kind = ("role_specific_public_position_v1" if positioned - {0} else expected[0]["kind"])
    require(header.get("kind") == kind and header.get("cancel_retry_guard") is True
            and header.get("guard_version") == 1, "Arena decoding header differs")
    covered, seen, candidate_events = set(), set(), []
    for group in report.get("position_decoders", []):
        members = group["roles"]
        require(members and all(type(role) is int and role in positioned for role in members),
                "Unexpected position decoder role")
        require(len(set(members)) == len(members) and not covered.intersection(members),
                "Position decoder roles overlap")
        require(all(expected[role] == expected[members[0]] for role in members),
                "One decoder is attributed to different frozen policies")
        covered.update(members)
        events = group["events"]
        require(group["statistics"]["moves"] == len(events), "Position Move counter differs")
        for event in events:
            check_event(event, identities=report["games"] * 8,
                        threshold=expected[members[0]]["threshold"], seen=seen)
            lane, seat = divmod(event["identity"], 8)
            role = (seat - lane) % 8
            require(role in members, "Position event belongs to another role")
            if role == 0:
                candidate_events.append(event)
        if 0 in members and "statistics" in report.get("position_decoder", {}):
            require(members == [0] and report["position_decoder"]["statistics"] == group["statistics"],
                    "Candidate counters include another role")
    require(covered == positioned, "Arena omits frozen position decoder evidence")
    require(("position_decoder" in report) == (0 in positioned), "Candidate position evidence differs")
    if 0 in positioned:
        require(report["position_decoder"]["events"] == candidate_events
                and report["position_decoder"]["counted_episode_moves"] == len(candidate_events),
                "Candidate position events differ from their role group")
