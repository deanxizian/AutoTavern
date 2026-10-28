"""Explicit policy-only transfer between engines with the same observation contract.

Record the source contract while its matching native extension is still installed.
Then rebuild, test the new rules and transfer trusted local checkpoints. This never
transfers optimizer, game, RNG or decoder state, and does not certify live compatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import taverngym as tg

from .common import atomic_json, make_policy


CONTRACT_FILES = (
    "crates/taverngym/src/actions.rs",
    "crates/taverngym/src/obs.rs",
    "crates/taverngym/src/cards.rs",
    "crates/taverngym/src/types.rs",
    "crates/taverngym/src/state.rs",
    "crates/taverngym-py/src/lib.rs",
    "training/common.py",
    "training/entity.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record_contract():
    if tg.build_is_current() is not True:
        raise ValueError("Record a contract only from a matching source checkout and native build")
    root = Path(__file__).resolve().parents[1]
    columns, static = tg.card_static()
    inputs = {
        "layout": dict(tg.layout()),
        "sizes": {key: getattr(tg, key) for key in ("N_ACTIONS", "N_CARDS", "OBS_FLOATS", "OBS_CARDS")},
        "actions": [tg.action_name(i) for i in range(tg.N_ACTIONS)],
        "card_ids": [None, *[tg.card_info(i)[0] for i in range(1, tg.N_CARDS)]],
        "card_static_columns": columns,
        "card_static_sha256": digest(static),
        "catalog_sha256": digest(tg.catalog_json().encode()),
        "source_sha256": {name: digest((root / name).read_bytes()) for name in CONTRACT_FILES},
    }
    return {"format_version": 1, "engine_hash": tg.build_info()["source_hash"], "inputs": inputs}


def validate_contract(source, target, checkpoint_engine):
    if source.get("format_version") != 1 or target.get("format_version") != 1:
        raise ValueError("Unsupported observation contract version")
    if source.get("engine_hash") != checkpoint_engine:
        raise ValueError("Source contract was not recorded for this checkpoint's engine")
    if not source.get("inputs") or source["inputs"] != target.get("inputs"):
        raise ValueError("Observation/card/action contract changed; policy transfer is not supported")


def transfer_checkpoint(source_path, source_contract, output_path, *, reason):
    source_path, output_path = Path(source_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError("Preserve existing checkpoints; choose a new output path")
    if not reason.strip():
        raise ValueError("Record the reason for the rules transition")
    target = record_contract()
    # These checkpoints contain pickle and must be trusted training artifacts.
    with source_path.open("rb") as handle:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
        checksum = checksum.hexdigest()
        handle.seek(0)
        old = torch.load(handle, map_location="cpu", weights_only=False)
    validate_contract(source_contract, target, old["engine_hash"])
    if old["engine_hash"] == target["engine_hash"]:
        raise ValueError("Both engines are identical; no rules transfer is needed")
    model = make_policy(old["architecture"])
    model.load_state_dict(old["model"], strict=True)
    if not all(torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
        raise ValueError("Checkpoint contains non-finite policy weights")
    # These are provenance fields, not a claim that old training ran on the new rules.
    keys = ("format_version", "training_code_hash", "architecture", "args", "model", "update", "steps")
    new = {key: old[key] for key in keys}
    new.update(engine_hash=target["engine_hash"], weights_only=True, phase="transferred_policy")
    new["weight_transfer"] = {
        "source_path": str(source_path.resolve()), "source_sha256": checksum,
        "source_engine_hash": old["engine_hash"], "target_engine_hash": target["engine_hash"],
        "source_update": old["update"], "source_steps": old["steps"], "reason": reason,
        "input_contract_sha256": digest(json.dumps(target["inputs"], sort_keys=True).encode()),
        "previous_transfer": old.get("weight_transfer"),
        "resume_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing a frozen opponent.
    with output_path.open("xb") as handle:
        torch.save(new, handle)
    return new["weight_transfer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("--out", type=Path, required=True)
    move = sub.add_parser("weights")
    move.add_argument("source", type=Path)
    move.add_argument("--source-contract", type=Path, required=True)
    move.add_argument("--out", type=Path, required=True)
    move.add_argument("--reason", required=True)
    args = parser.parse_args()
    if args.command == "record":
        if args.out.exists():
            parser.error("Preserve the existing source contract; choose a new path")
        contract = record_contract()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.out, contract)
        print({"out": str(args.out), "engine_hash": contract["engine_hash"]})
    else:
        provenance = transfer_checkpoint(args.source, json.loads(args.source_contract.read_text()),
                                         args.out, reason=args.reason)
        print({"out": str(args.out), **provenance})


if __name__ == "__main__":
    main()
