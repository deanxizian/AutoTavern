"""Label-free TavernBench adapter: choose candidate plans by PPO policy likelihood.

Only player-visible native observations enter the network. The simulator executes
listed recruit operations on copies; no combat search or value-head search is used.
"""

from collections import defaultdict
import json

import numpy as np
import torch

from tavernbench import engine as E


RULES = {"R1", "R3", "R4", "R5", "R6", "R7", "E1", "E2", "E4", "E5"}
COMBAT = {"R2", "B1", "B2", "B3", "B4", "B5", "B6"}


def observation(game, seat):
    features, cards = game.observe(seat)
    return {"features": np.frombuffer(features, dtype="<f4").copy(),
            "cards": np.frombuffer(cards, dtype="<i4").copy(),
            "action_mask": np.frombuffer(game.legal_mask(seat), dtype=np.uint8).astype(bool)}


def index(items, key, value):
    return next(i for i, item in enumerate(items) if item.get(key) == value)


def append(game, seat, trace, action, execute=True):
    obs = observation(game, seat)
    if not obs["action_mask"][action]:
        raise ValueError(f"illegal candidate action {action}")
    trace.append((action, obs))
    if execute:
        game.step(seat, action)


def operation(game, seat, trace, op):
    me = E.me(game, seat)
    kind = op["kind"]
    if kind in ("play", "play_target"):
        action = E.play_index(index(me["hand"], "eid", op["eid"]), len(me["board"]))
    elif kind == "spell":
        action = E.play_index(index(me["hand"], "hid", op["hid"]), 0)
    elif kind == "sell":
        action = E.A_SELL + index(me["board"], "eid", op["eid"])
    elif kind == "buy":
        action = E.A_BUY + index(me["shop"], "eid", op["eid"])
    elif kind == "choose":
        action = E.A_CHOOSE + op["k"]
    elif kind == "play_last":
        action = E.play_index(len(me["hand"]) - 1, len(me["board"]))
    elif kind == "spell_target":
        action = E.play_index(op["hand"], 0)
    elif kind == "activate_target":
        action = E.A_ACTIVATE + index(me["board"], "eid", op["eid"])
    elif kind == "power_target":
        action = E.A_HERO_POWER + op.get("slot", 0)
    elif kind == "magnetize":
        action = (E.A_MAGNETIZE + index(me["hand"], "eid", op["eid"]) * E.N_BOARD
                  + index(me["board"], "eid", op["target"]))
    else:
        raise ValueError(f"unknown operation {kind}")
    append(game, seat, trace, action)
    if kind == "buy" and op.get("play"):
        me = E.me(game, seat)
        hand = next((i for i, item in enumerate(me["hand"]) if item.get("eid") == op["eid"]),
                    len(me["hand"]) - 1)
        append(game, seat, trace, E.play_index(hand, len(me["board"])))
    if kind in ("spell_target", "play_target", "activate_target", "power_target"):
        target = op["target"] if kind == "play_target" else op.get("target", op.get("eid"))
        pending = E.me(game, seat)["pending"]
        if not pending or target not in pending.get("targets", []):
            raise ValueError("specified target is absent from the pending choice")
        append(game, seat, trace, E.A_CHOOSE + pending["targets"].index(target))
    if E.me(game, seat)["pending"] and not op.get("keep_pending"):
        raise ValueError("candidate opened an unspecified pending choice")


def candidates(scenario, spec, seed):
    seat = spec.get("seat", 0)
    if scenario == "D1":
        state = json.loads(spec["states"][0])
        heroes = [json.loads(value)["players"][0]["hero"] for value in spec["states"]]
        state["players"][0]["pending"] = [{"HeroSelect": {"options": heroes}}]
        base = E.reseeded(E.from_state(json.dumps(state)), seed)
        return [(answer, [(E.A_CHOOSE + i, observation(base, 0))])
                for i, answer in enumerate(spec["answers"])], []
    base = E.reseeded(E.from_state(spec["state"]), seed)
    if "map" in spec:
        obs = observation(base, seat)
        return [(answer, [(int(action), obs)]) for action, answer in spec["map"].items()
                if obs["action_mask"][int(action)]], []
    plans, invalid = [], []
    options = (list(zip(spec["answers"], spec["perms"])) if scenario.startswith("P") else
               [(item["answer"], item["ops"]) for item in spec["options"]])
    for answer, instructions in options:
        game, trace = base.copy(), []
        try:
            if scenario.startswith("P"):
                # One canonical left-to-right move sequence per listed arrangement.
                original = [item["eid"] for item in E.me(game, seat)["board"]]
                for dst, original_position in enumerate(instructions):
                    src = index(E.me(game, seat)["board"], "eid", original[original_position])
                    if src != dst:
                        append(game, seat, trace, E.move_index(src, dst))
                assert [item["eid"] for item in E.me(game, seat)["board"]] == [original[i] for i in instructions]
            else:
                for op in instructions:
                    operation(game, seat, trace, op)
            if scenario.startswith("P") or scenario == "E3":
                append(game, seat, trace, E.A_END, execute=False)
            plans.append((answer, trace))
        except (ValueError, StopIteration, KeyError) as exc:
            invalid.append({"answer": answer, "reason": f"{type(exc).__name__}: {exc}"})
    return plans, invalid


def choose(model, plans):
    """MAP plan under the policy, renormalized over legal candidate continuations.

    A required action shared by all remaining plans has probability one. Nodes are
    evaluated once, in batches; this evaluates policy likelihood, not combat value.
    """
    branches, observations = defaultdict(set), {}
    for _, trace in plans:
        prefix = ()
        for action, obs in trace:
            branches[prefix].add(action)
            observations.setdefault(prefix, obs)
            prefix += (action,)
    branching = [prefix for prefix, actions in branches.items() if len(actions) > 1]
    log_probs = {}
    for begin in range(0, len(branching), 128):
        prefixes = branching[begin:begin + 128]
        batch = {key: np.stack([observations[prefix][key] for prefix in prefixes])
                 for key in ("features", "cards", "action_mask")}
        masks = np.zeros_like(batch["action_mask"])
        for i, prefix in enumerate(prefixes):
            masks[i, list(branches[prefix])] = True
        assert not (masks & ~batch["action_mask"]).any()
        with torch.inference_mode():
            obs, _ = model.policy.obs_to_tensor(batch)
            distribution = model.policy.get_distribution(obs, action_masks=masks)
            values = distribution.distribution.logits.cpu().numpy()
        for prefix, row in zip(prefixes, values):
            log_probs[prefix] = row
    scores = []
    for answer, trace in plans:
        prefix, score = (), 0.0
        for action, _ in trace:
            if prefix in log_probs:
                score += float(log_probs[prefix][action])
            prefix += (action,)
        scores.append({"answer": answer, "log_probability": score,
                       "actions": [action for action, _ in trace]})
    selected = max(scores, key=lambda item: item["log_probability"])
    return {"answer": selected["answer"]}, {"selected": selected, "candidates": scores,
                                             "policy_nodes": len(branching)}


def predict(model, scenario, spec, seed):
    if scenario in RULES or scenario in COMBAT:
        return {"status": "unsupported_question", "response": None,
                "reason": "no rule-QA interface" if scenario in RULES else "no combat prediction head"}
    plans, invalid = candidates(scenario, spec, seed)
    if not plans:
        return {"status": "unmappable_task", "response": None,
                "reason": "no public candidate can be executed under the saved state's legal mask",
                "details": {"invalid_candidates": invalid}}
    response, details = choose(model, plans)
    return {"status": "ok", "response": response,
            "details": {**details, "invalid_candidates": invalid}}
