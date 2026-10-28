"""Detect valid-looking placements produced by protective simulator cutoffs."""

import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import taverngym as tg
from training.common import Policy, evaluate
from training.diagnostics import (BehaviorAudit, SimulatorGuardError, check_game_guards,
                                  check_seed_range, replay_terminal_step)


@pytest.mark.parametrize("controlled", [(0,), (0, 3), tuple(range(8))])
def test_terminal_replay_matches_native_batch(controlled):
    env = tg.BatchEnv(1, seed=82, controlled=controlled, config={"turn_action_caps": [32]})
    completed = 0
    for _ in range(1500):
        before = env.game(0)
        obs = env.observe()
        actions = [before.bot_action(seat) if obs["ready"][j] else 0
                   for j, seat in enumerate(controlled)]
        _, done, info = env.step(actions)
        if done.any():
            replay = replay_terminal_step(before, actions, controlled, done, info["placement"],
                                          context={"seed": 82})
            if not done.all() and env.episode_index(0) == 0:
                assert replay.state_json() == env.game(0).state_json()
            completed += int(done.sum())
        if env.episode_index(0) > 0:
            break
    assert completed == len(controlled)


@pytest.mark.parametrize("field,value", [("truncated_combats", 1), ("turn_limit_end", True)])
def test_guard_evidence_survives_valid_placements(field, value):
    state = json.loads(tg.Game(4).state_json())
    state[field] = value
    game = tg.Game.from_json(json.dumps(state))
    with pytest.raises(SimulatorGuardError) as failure:
        check_game_guards(game, context={"lane": 2})
    assert failure.value.evidence[field] == value


def test_single_seat_audit_keeps_same_scores():
    torch.set_num_threads(2)
    model = Policy(hidden=8, embedding=2)
    args = dict(games=8, seed=90, threads=1, config={"turn_action_caps": [32]},
                device=torch.device("cpu"), kind="bot")
    plain = evaluate(model, **args)
    audited = evaluate(model, audit_guards=True, **args)
    assert plain["placements_by_seat"] == audited["placements_by_seat"]
    assert not plain["guards_checked"] and plain["truncated_combats"] is None
    assert audited["guards_checked"] and audited["truncated_combats"] == 0


def test_seed_ranges_cannot_partially_overlap():
    check_seed_range(100, 8, 10, 20)
    for base in [3, 10, 13, 20, 27]:
        with pytest.raises(ValueError, match="overlaps"):
            check_seed_range(base, 8, 10, 20)


def test_behavior_counts_do_not_mix_seats_or_turns():
    audit = BehaviorAudit()
    obs = {"features": np.array([[.05], [.05]], np.float32),
           "cards": np.zeros((2, 1), np.int32)}
    audit.record(obs, [0, 0], [0, 1])
    audit.record(obs, [0, 0], [0])
    obs["features"][0, 0] = .1
    audit.record(obs, [0, 0], [0])
    # Another evaluation seat reuses lane zero but is a different identity.
    audit.record(obs, [0, 0], [0], identity_offset=8)
    report = audit.report()
    assert report["repeated_visible_observations"] == 1
    assert report["decisions"] == 5
    assert report["recruit_turns"] == 4
