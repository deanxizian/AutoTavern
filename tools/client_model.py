"""Local trained-policy inference, with explicit client-input completeness checks.

Run with the dedicated model environment; importing the log reader never imports
Torch. No endpoint sends input to Hearthstone. All outputs are development evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .client_common import write_private_json
from .client_actions import render_bound_action
from .client_features import ClientFeatureAdapter, ClientInput


ROOT = Path(__file__).resolve().parents[1]


class ModelUnavailable(ValueError):
    pass


@dataclass
class PolicyResult:
    action: int
    probability: float
    value: float
    latency_ms: float


class PolicyRuntime:
    def __init__(self, checkpoint, *, expected_sha256=None, threads=4, position_bundle=None,
                 position_sha256=None, cancel_retry_guard=True):
        import torch
        import taverngym as tg
        from training.common import load_checkpoint
        if type(threads) is not int or not 1 <= threads <= 32:
            raise ModelUnavailable("Invalid inference thread count")
        self.path = Path(checkpoint).resolve(strict=True)
        if not self.path.is_file():
            raise ModelUnavailable("Select a trainer checkpoint file")
        before = self.path.stat()
        torch.set_num_threads(threads)
        started = time.perf_counter()
        with self.path.open("rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            self.sha256 = digest.hexdigest()
            if expected_sha256 is not None and self.sha256 != expected_sha256:
                raise ModelUnavailable("Checkpoint checksum changed")
            if tg.build_is_current() is not True:
                raise ModelUnavailable("Native build does not match this checkout")
            handle.seek(0)
            self.policy, metadata = load_checkpoint(handle, "cpu")
        self.position = position_bundle is not None
        self.client_game_key = None
        if position_bundle is not None:
            from training.position import load_position_policy
            bundle = Path(position_bundle).resolve(strict=True)
            checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
            if position_sha256 is not None and checksum != position_sha256:
                raise ModelUnavailable("Position bundle checksum changed")
            actor_info = {"sha256": self.sha256, "engine_hash": metadata["engine_hash"],
                          "action_cap": metadata["args"]["action_cap"],
                          "trained_hotfix_37_0_1": metadata["args"].get("hotfix_37_0_1", False)}
            self.policy = load_position_policy(self.policy, actor_info, bundle, "cpu")
        from training.decoding import CancelRetryGuard
        self.guard = CancelRetryGuard() if cancel_retry_guard else None
        after = self.path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ModelUnavailable("Checkpoint changed during loading; use a completed immutable artifact")
        self.policy.eval().requires_grad_(False)
        if not all(torch.isfinite(t).all() for t in self.policy.state_dict().values() if t.is_floating_point()):
            raise ModelUnavailable("Non-finite model weights")
        self.metadata = {"checkpoint": str(self.path), "sha256": self.sha256,
            "engine_hash": metadata["engine_hash"], "architecture": metadata["architecture"],
            "update": metadata.get("update"), "phase": metadata.get("phase"),
            "strategy_sha256": metadata.get("strategy_sha256"), "device": "cpu", "threads": threads,
            "position": getattr(self.policy, "decoder_metadata", None),
            "cancel_retry_guard": bool(self.guard),
            "requirements": {name: bool(getattr(self.policy, "requires_" + name, False)) for name in ("attachments", "growth", "strategy")},
            "load_ms": round((time.perf_counter() - started) * 1000, 3),
            "model_ready": False, "live_ready": False}
        # Validate ordered vocabulary, not just the number of embeddings.
        self.adapter = ClientFeatureAdapter(self.metadata["requirements"])
        actual = {tg.card_info(i)[0]: i for i in range(1, tg.N_CARDS)}
        if actual != self.adapter.vocabulary or (tg.OBS_FLOATS, tg.OBS_CARDS, tg.N_ACTIONS) != (3227, 124, 245):
            raise ModelUnavailable("Client adapter observation/card/action layout differs")

    def observe_client_history(self, snapshot):
        source = snapshot.get("source", {})
        key = tuple(source.get(k) for k in ("path", "device", "inode", "game_start_line"))
        if key != self.client_game_key:
            if self.position: self.policy.reset_context(0)
            if self.guard: self.guard.reset(0)
            self.client_game_key = key
        move = snapshot.get("public_history", {}).get("confirmed_move")
        if self.position and move and isinstance(move.get("turn"), int):
            self.policy.load_context({"version": self.policy.version,
                "states": [{"identity": 0, "turn": move["turn"], "moved": True}]}, identities=[0])

    def infer_arrays(self, observation, *, strategy_view=None):
        """Speculate without committing position or cancellation history."""
        from training.position import preserve_position_context
        guard_state = self.guard.save(0) if self.guard else None
        try:
            with preserve_position_context([self.policy]) if self.position else nullcontext():
                return self._infer_arrays(observation, strategy_view=strategy_view)
        finally:
            if self.guard:
                self.guard.load(0, guard_state)

    def _infer_arrays(self, observation, *, strategy_view=None):
        import torch
        import taverngym as tg
        from training.attachments import policy_kwargs
        started = time.perf_counter()
        expected = {"features": ((1, tg.OBS_FLOATS), np.float32), "cards": ((1, tg.OBS_CARDS), np.int32),
                    "action_mask": ((1, tg.N_ACTIONS), np.bool_)}
        for key, (shape, dtype) in expected.items():
            value = observation.get(key)
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
                raise ModelUnavailable("Invalid model input: " + key)
        if not np.isfinite(observation["features"]).all() or not observation["action_mask"].any():
            raise ModelUnavailable("Incomplete features or empty legal-action mask")
        if np.any(observation["cards"] < 0) or np.any(observation["cards"] >= tg.N_CARDS):
            raise ModelUnavailable("Unknown model card index")
        observation = dict(observation)
        if self.guard:
            observation = self.guard.apply({**observation, "ready": np.ones(1, dtype=np.bool_)}, identities=[0])
        if getattr(self.policy, "requires_strategy", False):
            if strategy_view is None:
                raise ModelUnavailable("Missing complete public strategy view")
            observation["strategy_bias"] = np.asarray([
                self.policy.teacher.advise(strategy_view, observation["action_mask"][0])], dtype=np.float32)
        with torch.inference_mode():
            args = [torch.from_numpy(np.array(observation[k], copy=True)) for k in ("features", "cards", "action_mask")]
            if self.position:
                self.policy.set_context([0])
            logits, value = self.policy(*args, **policy_kwargs(self.policy, observation, "cpu"))
            if logits.shape != (1, tg.N_ACTIONS) or value.numel() != 1 or not torch.isfinite(logits).all() or not torch.isfinite(value).all():
                raise ModelUnavailable("Invalid or non-finite model output")
            # Mask a second time at the boundary; even an erroneous actor cannot
            # return an unmapped action. Empty masks were rejected above.
            logits = logits.masked_fill(~args[2], -torch.inf)
            action = int(logits.argmax(-1).item())
            probability = float(logits.softmax(-1)[0, action].item())
        return PolicyResult(action, probability, float(value.item()), (time.perf_counter() - started) * 1000)

    def infer_client(self, packet):
        if not isinstance(packet, ClientInput) or packet.scope != "client_recording" or not packet.complete:
            raise ModelUnavailable("Client observation is incomplete")
        observation = {k: np.array(getattr(packet, k), copy=True)[None, :] for k in ("features", "cards", "action_mask")}
        observation.update(attachments=packet.attachments, growth=packet.growth)
        from .client_decisions import pending_observation
        mapping = deepcopy(packet.actions)
        skipped, total_ms = [], 0.0
        for _ in range(int(packet.action_mask.sum())):
            result = self.infer_arrays(observation, strategy_view=packet.strategy_view)
            total_ms += result.latency_ms
            binding = mapping.bindings[result.action]
            if binding.get("targets") or binding.get("modes"):
                try:
                    second, view, candidates = pending_observation(packet, binding, self.adapter.vocabulary)
                    if view.get("continuation_projection"):
                        binding["continuation_projection"] = view["continuation_projection"]
                    decision = self.infer_arrays(second, strategy_view=view)
                    total_ms += decision.latency_ms
                    if decision.action == 7: raise ValueError("Model cancelled target selection")
                    chosen = candidates[decision.action-30]
                    binding["resolved_mode" if binding.get("modes") else "resolved_target"] = chosen
                    if binding.get("modes"):
                        if any(c["kind"] != "target" for c in chosen["children"]):
                            raise ValueError("Unsupported nested choice menu")
                        binding["targets"] = [c["entity_id"] for c in chosen["children"]
                                              if c.get("error") is None]
                        if binding["targets"]:
                            target_binding = {k: v for k, v in binding.items() if k != "modes"}
                            third, view, targets = pending_observation(packet, target_binding, self.adapter.vocabulary)
                            target_decision = self.infer_arrays(third, strategy_view=view)
                            total_ms += target_decision.latency_ms
                            if target_decision.action == 7:
                                raise ValueError("Model cancelled target selection")
                            binding["resolved_target"] = targets[target_decision.action-30]
                except ValueError as error:
                    skipped.append({"action": result.action, "reason": str(error)})
                    observation["action_mask"][0, result.action] = False
                    if not observation["action_mask"].any(): raise ModelUnavailable("No fully resolved action")
                    continue
            instruction = render_bound_action(mapping, result.action)
            break
        return {**instruction, "policy": self.metadata, "probability": result.probability,
                "value": result.value, "inference_ms": round(total_ms, 3), "excluded_actions": skipped,
                "scope": "client_live_advice", "model_ready": True, "live_ready": True}


def native_observation(game, seat, runtime):
    """Test/benchmark source only. Never replace a client input with a new game."""
    from training.attachments import AttachmentRows
    from training.growth import from_views
    f, c = game.observe(seat)
    view = json.loads(game.strategy_view(seat))
    observation = {"features": np.frombuffer(f, dtype="<f4").copy()[None, :],
                   "cards": np.frombuffer(c, dtype="<i4").copy()[None, :],
                   "action_mask": np.frombuffer(game.legal_mask(seat), dtype=np.bool_).copy()[None, :]}
    if runtime.metadata["requirements"]["attachments"]:
        observation["attachments"] = AttachmentRows.from_games([game], [(0, seat)])
    if runtime.metadata["requirements"]["growth"]:
        observation["growth"] = from_views([view])
    return observation, view


def benchmark(runtime, count=32):
    """Representative hero/recruit/choice states; not a strength assessment."""
    import taverngym as tg
    game = tg.Game(739175, json.dumps({"hotfix_37_0_1": True, "turn_action_caps": [128]}))
    rows = []
    start = time.perf_counter()
    for step in range(4000):
        for seat in range(8):
            mask = np.frombuffer(game.legal_mask(seat), dtype=np.bool_)
            if not mask.any():
                continue
            if seat == 0 and len(rows) < count:
                obs, view = native_observation(game, seat, runtime)
                result = runtime.infer_arrays(obs, strategy_view=view)
                # Infer twice without advancing: output must not mutate a decoder
                # or acquire hidden private game state.
                repeated = runtime.infer_arrays(obs, strategy_view=view)
                if result.action != repeated.action or result.probability != repeated.probability:
                    raise ModelUnavailable("Inference is not repeatable on an unchanged observation")
                rows.append({"turn": view["turn"], "phase": view["phase"],
                    "pending": (view["me"].get("pending") or {}).get("kind"),
                    "action": result.action, "first_ms": round(result.latency_ms, 3), "warm_ms": round(repeated.latency_ms, 3)})
            # A scripted driver collects diverse inputs, including complete turns.
            # Benchmarking policy latency is separate from evaluating its strategy.
            game.step(seat, game.bot_action(seat))
        if len(rows) >= count:
            break
        if not any(np.frombuffer(game.legal_mask(p), dtype=np.bool_).any() for p in range(8)):
            break
    if not rows:
        raise ModelUnavailable("No benchmark decisions collected")
    times = np.asarray([r["warm_ms"] for r in rows])
    return {"schema": "autotavern.model-benchmark.v1", "scope": "simulator_inputs_only",
            "policy": runtime.metadata, "decisions": len(rows), "requested_decisions": count,
            "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
            "max_ms": float(times.max()), "elapsed_seconds": round(time.perf_counter() - start, 3),
            "rows": rows, "model_ready": False, "live_ready": False}


def replay(runtime, directory):
    rows, issues, timings = [], Counter(), []
    for path in sorted(Path(directory).glob("snapshot-*.json.gz")):
        snapshot = json.loads(gzip.decompress(path.read_bytes()))
        start = time.perf_counter()
        packet = runtime.adapter.convert(snapshot)
        timings.append((time.perf_counter() - start) * 1000)
        report = packet.report()
        issues.update({g["code"] for g in packet.gaps})
        row = {"snapshot": path.name, **report}
        if packet.complete:
            try:
                row["decision"] = runtime.infer_client(packet)
            except (ModelUnavailable, ValueError) as error:
                row["inference_error"] = str(error)
        rows.append(row)
    if not rows:
        raise ModelUnavailable("No recorded snapshots found")
    return {"schema": "autotavern.model-replay.v1", "recording": str(Path(directory).resolve()),
            "policy": runtime.metadata, "frames": len(rows),
            "complete_inputs": sum(r["status"] == "complete" for r in rows),
            "inferred_frames": sum("decision" in r for r in rows), "gap_counts": dict(issues),
            "adapter_median_ms": float(np.median(timings)), "adapter_p95_ms": float(np.percentile(timings, 95)),
            "rows": rows, "model_ready": False, "live_ready": False}


def worker(runtime):
    """Private stdin/stdout protocol, one current frame at a time, no network."""
    print(json.dumps({"status": "loaded", "policy": runtime.metadata}), flush=True)
    for line in sys.stdin:
        packet = None
        message = None
        try:
            message = json.loads(line)
            runtime.observe_client_history(message["snapshot"])
            packet = runtime.adapter.convert(message["snapshot"])
            report = packet.report()
            result = {"frame_id": packet.frame_id, "input": report, "proposal": None,
                      "policy": runtime.metadata, "status": "input_incomplete"}
            if packet.complete:
                result["proposal"] = runtime.infer_client(packet)
                result["status"] = "inferred"
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except Exception as error:
            # No traceback containing private raw entities is sent to the UI.
            from .client_actions import frame_identity
            frame = packet.frame_id if packet else frame_identity(message["snapshot"]) if isinstance(message, dict) and "snapshot" in message else None
            print(json.dumps({"frame_id": frame, "status": "failed", "error_type": type(error).__name__}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "benchmark", "replay", "worker"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--position-bundle", type=Path)
    parser.add_argument("--position-sha256")
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--decisions", type=int, default=32)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "replay" and args.recording is None:
        parser.error("replay requires --recording")
    if args.decisions < 1 or args.decisions > 1024:
        parser.error("decisions must be in 1..1024")
    runtime = PolicyRuntime(args.checkpoint, expected_sha256=args.sha256, threads=args.threads,
                            position_bundle=args.position_bundle, position_sha256=args.position_sha256)
    if args.command == "worker":
        worker(runtime)
        return
    report = (runtime.metadata if args.command == "inspect" else benchmark(runtime, args.decisions)
              if args.command == "benchmark" else replay(runtime, args.recording))
    if args.out:
        write_private_json(args.out, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
