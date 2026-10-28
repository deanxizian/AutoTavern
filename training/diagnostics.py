"""Check terminal transitions that BatchEnv normally hides behind auto-reset."""

from collections import Counter
import hashlib
import json

import numpy as np


class BehaviorAudit:
    """Descriptive operation counts; a repeated observation is not itself a failure."""
    def __init__(self):
        self.active = {}
        self.finished_turns = []
        self.actions = Counter()
        self.repeats = 0

    def record(self, obs, actions, slots, identity_offset=0):
        for slot in slots:
            turn = round(float(obs["features"][slot, 0]) * 20)
            key = identity_offset + int(slot)
            prior = self.active.get(key)
            if prior is None or prior[0] != turn:
                if prior is not None and prior[0] > 0:
                    self.finished_turns.append(prior[1])
                prior = self.active[key] = [turn, 0, set()]
            digest = hashlib.sha256(obs["features"][slot].tobytes()
                                    + obs["cards"][slot].tobytes()).digest()
            self.repeats += int(digest in prior[2])
            prior[2].add(digest)
            prior[1] += 1
            self.actions[int(actions[slot])] += 1

    def report(self):
        import taverngym as tg
        turns = self.finished_turns + [row[1] for row in self.active.values() if row[0] > 0]
        return {
            "decisions": sum(self.actions.values()),
            "repeated_visible_observations": self.repeats,
            "recruit_turns": len(turns),
            "turns_at_least_128_decisions": sum(n >= 128 for n in turns),
            "decisions_per_recruit_turn_p50_p95_max": (
                np.percentile(turns, [50, 95, 100]).tolist() if turns else None),
            "action_counts": {tg.action_name(a): n for a, n in sorted(self.actions.items())},
        }


class SimulatorGuardError(RuntimeError):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__("Simulator guard invalidates evaluation: " + json.dumps(evidence))


def check_game_guards(game, *, context):
    state = json.loads(game.state_json())
    evidence = {**context, "turn": game.turn,
                "truncated_combats": game.truncated_combats,
                "turn_limit_end": game.turn_limit_end,
                "stopped_reason": state.get("stopped_reason")}
    if evidence["truncated_combats"] or evidence["turn_limit_end"] or evidence["stopped_reason"]:
        raise SimulatorGuardError(evidence)


def replay_terminal_step(game, actions, controlled, done, placement, *, context):
    """Replay a BatchEnv step on its pre-step copy, retaining the terminal game.

    This mirrors BatchEnv::step_game/advance without its auto-reset. Ready seats are
    captured before actions are applied, and bot seats act in ascending seat order.
    No engine or card rules are modified. Placements must match the native transition.
    """
    ready = [game.needs_action(seat) for seat in controlled]
    for j, seat in enumerate(controlled):
        if ready[j] and not game.is_over() and game.needs_action(seat):
            game.step(seat, int(actions[j]))
    for _ in range(10_000):
        if game.is_over():
            break
        progressed = False
        for seat in range(game.num_players):
            if seat not in controlled and game.needs_action(seat):
                game.bot_play(seat)
                progressed = True
        if any(game.needs_action(seat) for seat in controlled):
            break
        if not progressed:
            # The native driver would mark NoProgress unless the game just ended.
            if not game.is_over():
                raise SimulatorGuardError({**context, "stopped_reason": "replay_no_progress"})
            break
    else:
        raise SimulatorGuardError({**context, "stopped_reason": "replay_driver_limit"})
    actual = game.placements()
    for j, seat in enumerate(controlled):
        if done[j] and actual[seat] != int(placement[j]):
            raise RuntimeError(f"Terminal replay differs from BatchEnv: {context}, seat {seat}")
    check_game_guards(game, context=context)
    return game


def check_seed_range(seed, count, training_seed, validation_seed):
    if count < 1 or seed < 0 or seed + count > 2**64:
        raise ValueError("Invalid evaluation seed range")
    used = {training_seed, *range(validation_seed, validation_seed + 8)}
    overlap = sorted(used.intersection(range(seed, seed + count)))
    if overlap:
        raise ValueError(f"Evaluation overlaps training/validation seed(s): {overlap}")
