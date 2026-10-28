"""A rules update may reuse a policy, but must not silently resume old game state."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import taverngym as tg

from training.common import Policy, load_checkpoint, tensors
from training.transfer import record_contract, transfer_checkpoint, validate_contract


@pytest.fixture
def source(tmp_path):
    torch.manual_seed(37)
    model = Policy(hidden=16, embedding=2)
    checkpoint = {
        "format_version": 1, "engine_hash": "previous-engine", "training_code_hash": "old-code",
        "architecture": model.architecture, "args": {"seed": 37, "eval_seed": 97, "action_cap": 128},
        "model": model.state_dict(), "steps": 1234, "update": 5,
        "optimizer": {"must_not_transfer": True}, "environment": b"old game state",
        "torch_rng": torch.get_rng_state(), "anchor": model.state_dict(),
    }
    path = tmp_path / "source.pt"
    torch.save(checkpoint, path)
    contract = record_contract()
    contract["engine_hash"] = checkpoint["engine_hash"]
    return path, contract, model


def test_transfer_preserves_exact_policy_and_provenance_but_no_resume_state(source, tmp_path):
    path, contract, model = source
    output = tmp_path / "transferred.pt"
    provenance = transfer_checkpoint(path, contract, output, reason="Audited server hotfix")
    loaded, checkpoint = load_checkpoint(output)
    assert checkpoint["weights_only"] is True
    assert not {"optimizer", "environment", "anchor", "torch_rng", "cuda_rng", "numpy_rng"} & checkpoint.keys()
    assert provenance["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance["source_engine_hash"] == "previous-engine"
    assert provenance["target_engine_hash"] == tg.source_hash()
    assert provenance["source_steps"] == 1234
    assert provenance["resume_allowed"] is False
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])
    env = tg.BatchEnv(8, seed=710, threads=1)
    f, c, m = tensors(env.observe(), torch.device("cpu"))
    with torch.inference_mode():
        original, moved = model(f, c, m), loaded(f, c, m)
    assert all(torch.equal(a, b) for a, b in zip(original, moved))
    with pytest.raises(FileExistsError):
        transfer_checkpoint(path, contract, output, reason="Must not overwrite")


@pytest.mark.parametrize("change", ["card_order", "action_order", "feature_meaning", "static_values"])
def test_transfer_rejects_semantic_changes_even_when_dimensions_match(source, tmp_path, change):
    path, contract, _ = source
    bad = copy.deepcopy(contract)
    inputs = bad["inputs"]
    if change == "card_order":
        inputs["card_ids"][1], inputs["card_ids"][2] = inputs["card_ids"][2], inputs["card_ids"][1]
    elif change == "action_order":
        inputs["actions"][0], inputs["actions"][1] = inputs["actions"][1], inputs["actions"][0]
    elif change == "feature_meaning":
        inputs["source_sha256"]["crates/taverngym/src/obs.rs"] = "changed semantics"
    else:
        inputs["card_static_sha256"] = "changed values"
    output = tmp_path / "refused.pt"
    with pytest.raises(ValueError, match="contract changed"):
        transfer_checkpoint(path, bad, output, reason="Must reject")
    assert not output.exists()


def test_contract_must_belong_to_checkpoint_engine(source):
    _, contract, _ = source
    with pytest.raises(ValueError, match="checkpoint's engine"):
        validate_contract(contract, record_contract(), "unrelated-engine")


def test_record_refuses_stale_native_extension(monkeypatch):
    monkeypatch.setattr(tg, "build_is_current", lambda: False)
    with pytest.raises(ValueError, match="matching source"):
        record_contract()


def test_transfer_rejects_nonfinite_weights(source, tmp_path):
    path, contract, _ = source
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["model"]["actor.weight"][0, 0] = float("nan")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="non-finite"):
        transfer_checkpoint(path, contract, tmp_path / "bad.pt", reason="Must reject")


def test_transferred_policy_cannot_be_used_as_exact_resume(source, tmp_path):
    path, contract, _ = source
    output = tmp_path / "transferred.pt"
    transfer_checkpoint(path, contract, output, reason="Resume safety test")
    result = subprocess.run([
        sys.executable, "-m", "training.train", "--run-dir", str(tmp_path / "run"),
        "--resume", str(output), "--device", "cpu", "--envs", "1",
        "--model", "mlp", "--hidden", "16", "--embedding", "2", "--warmup-samples", "0",
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "Transferred policy has no training state" in result.stderr


def test_full_resume_refuses_changed_rules_before_restoring_environment(source, tmp_path):
    path, _, _ = source
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["engine_hash"] = tg.source_hash()
    torch.save(checkpoint, path)
    result = subprocess.run([
        sys.executable, "-m", "training.train", "--run-dir", str(tmp_path / "run"),
        "--resume", str(path), "--hotfix-37-0-1", "--device", "cpu", "--envs", "1",
        "--model", "mlp", "--hidden", "16", "--embedding", "2", "--warmup-samples", "0",
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "Resume changes the hotfix rules" in result.stderr


@pytest.mark.parametrize("module_name", ["training.evaluate", "training.arena"])
@pytest.mark.parametrize("override,expected", [(None, True), (True, True), (False, False)])
def test_evaluation_keeps_candidate_rules_unless_explicitly_overridden(
        source, tmp_path, monkeypatch, module_name, override, expected):
    path, _, _ = source
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["engine_hash"] = tg.source_hash()
    checkpoint["args"]["hotfix_37_0_1"] = True
    torch.save(checkpoint, path)
    module = importlib.import_module(module_name)
    def capture(*args, **kwargs):
        return {"rules": kwargs["config"], "candidate": {"games": 8}, "elapsed_seconds": 0.0}
    monkeypatch.setattr(module, "play_arena" if module_name.endswith("arena") else "evaluate", capture)
    output = tmp_path / "evaluation.json"
    argv = [module_name, str(path), "--games", "8", "--seed", "910000003",
            "--device", "cpu", "--out", str(output)]
    if override is not None:
        argv.append("--hotfix-37-0-1" if override else "--no-hotfix-37-0-1")
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    report = json.loads(output.read_text())
    assert report["rules"].get("hotfix_37_0_1", False) is expected
