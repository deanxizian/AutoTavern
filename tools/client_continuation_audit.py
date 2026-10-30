"""Exercise every legal continuation, including branches the actor never picks."""
from copy import deepcopy
from .client_decisions import pending_observation
from .client_actions import render_bound_action


def audit(packet, vocabulary, runtime=None):
    report = {"legal_actions": len(packet.actions.bindings), "stages": 0,
        "battlecry_stages": 0, "random_projections": 0, "errors": []}
    if not packet.complete:
        report["errors"].append({"reason": "incomplete_input", "gaps": packet.gaps})
        return report
    for action, original in packet.actions.bindings.items():
        binding = deepcopy(original)
        try:
            if not binding.get("targets") and not binding.get("modes"):
                render_bound_action(packet.actions, action)
                continue
            def check(b):
                obs, view, candidates = pending_observation(packet, b, vocabulary)
                if runtime:
                    result = runtime.infer_arrays(obs, strategy_view=view)
                    assert obs["action_mask"][0, result.action]
                report["stages"] += 1
                report["battlecry_stages"] += b["kind"] == "play_minion"
                report["random_projections"] += bool(view.get("continuation_projection", {}).get("random_transition"))
                return candidates
            candidates = check(binding)
            if binding.get("modes"):
                for mode in candidates:
                    targets = [c["entity_id"] for c in mode.get("children", []) if c["kind"] == "target" and c.get("error") is None]
                    if targets:
                        branch = {k: v for k, v in binding.items() if k != "modes"}
                        branch.update(resolved_mode=mode, targets=targets)
                        check(branch)
        except (ValueError, AssertionError, KeyError, IndexError) as error:
            report["errors"].append({"action": action, "kind": binding["kind"],
                "card_id": binding.get("card_id"), "reason": str(error)})
    return report
