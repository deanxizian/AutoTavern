"""Generation-specific termination must affect only the actual triggering seats."""

import numpy as np
import pytest
import json
import taverngym as tg

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")

from taverngym.learning.actor_sampling import ActorOutput, UnifiedActorSampler
from taverngym.learning.opponents import FrozenOpponentPool, OpponentGroup
from taverngym.learning.vector_env import MixedOpponentVectorEnv


LEGACY = {"operation_cost": 0, "cycle_cost": .0002, "cycle_detection": "all_actions"}
CURRENT = {**LEGACY, "cycle_forfeit_repetitions": 3, "turn_action_forfeit_limit": 60}


class FreezeActor:
    def act(self, obs):
        actions = obs["action_mask"].argmax(axis=1)
        actions[obs["action_mask"][:, 2]] = 2
        return ActorOutput(actions, np.zeros(len(actions), np.float32),
                           np.zeros(len(actions), np.float32))

    def predict(self, obs, deterministic=False):
        return self.act(obs).actions, None


def pool(protocol=CURRENT, *, old=0, new=2):
    groups = []
    if old:
        groups.append(OpponentGroup("bot_v3", FreezeActor(), old))
    groups.append(OpponentGroup("bot_v4", FreezeActor(), new, protocol))
    return FrozenOpponentPool(tuple(groups))


def sampler(*, games=1, learner=LEGACY, protocol=CURRENT, old=0, new=2, seed=12):
    return UnifiedActorSampler(games, FreezeActor(), pool(protocol, old=old, new=new),
        seed=seed, threads=1, config={"turn_action_caps": [60], "freeze_presses_per_turn": 0,
            "hero_pool": ["TB_BaconShop_HERO_34"], "hero_choices": 1},
        frozen_opponents=old + new, reward_shaping=learner)


def test_opponent_only_cycle_deaths_keep_learner_and_old_generation_alive_across_rollouts():
    env = sampler(old=1)
    env.collect(6)  # Hero choice plus five Freeze actions; no complete third loop yet.
    assert env.opponent_events == [[]]
    seats = env.batch.slot_seats.copy()
    env.collect(1)
    events = env.opponent_events[0]
    assert len(events) == 2
    assert {event["role"] for event in events} == {2, 3}
    assert {event["reason"] for event in events} == {"cycle_forfeit"}
    assert {event["placement"] for event in events} == {7, 8}
    ordered = sorted(events, key=lambda event: event["physical_seat"])
    assert [event["placement"] for event in ordered] == [7, 8]
    assert env.batch.episode_index(0) == 0
    assert env.episodes[0] == 0
    state = __import__("json").loads(env.batch.save(0))
    assert state["game"]["players"][seats[0]]["alive"]
    assert state["game"]["players"][seats[1]]["alive"]
    assert all(not state["game"]["players"][seats[role]]["alive"] for role in (2, 3))


@pytest.mark.parametrize("mode,limit", [("cycle", 60), ("budget", 4), ("budget", 60)])
def test_simultaneous_learner_and_opponents_use_physical_seat_ties(mode, limit):
    costs = (CURRENT if mode == "cycle" else
             {**LEGACY, "turn_action_forfeit_limit": limit})
    env = sampler(games=8, learner=costs, protocol=costs)
    seats = env.batch.slot_seats.copy().reshape(8, 3)
    before_terminal = 6 if mode == "cycle" else limit + 1
    env.collect(before_terminal)
    transitions = []
    env.collect(1, transitions.append)
    assert len(transitions) == 8 and all(t.done for t in transitions)
    for transition in transitions:
        tie_order = sorted(range(3), key=lambda role: seats[transition.game, role])
        expected = {role: 6 + rank for rank, role in enumerate(tie_order)}
        assert transition.placement == expected[0]
        events = transition.info["opponent_terminations"]
        assert {event["role"]: event["placement"] for event in events} == {
            role: expected[role] for role in (1, 2)}
        assert transition.info["termination_reason"] == (
            "cycle_forfeit" if mode == "cycle" else "action_limit_forfeit")
        assert transition.info["episode"]["l"] == before_terminal + 1
        assert env.batch.episode_index(transition.game) == 1
    assert len({t.placement for t in transitions}) > 1


def test_last_opponents_forfeit_gives_learner_first_once_on_a_real_action():
    protocol = {**CURRENT, "cycle_forfeit_repetitions": 1}
    env = sampler(protocol=protocol, new=7)
    transitions = []
    result = env.collect(3, transitions.append)
    assert result["learner_samples"] == len(transitions) == 3
    assert [t.done for t in transitions] == [False, False, True]
    terminal = transitions[-1]
    assert terminal.action == 2 and terminal.placement == 1
    assert terminal.reward == pytest.approx(1 - .0002)
    assert "termination_reason" not in terminal.info
    assert len(terminal.info["opponent_terminations"]) == 7
    assert terminal.info["reward_components"]["placement_reward"] == 1
    assert terminal.info["reward_components"]["opponent_cycle_forfeits"] == 7
    env.collect(1, transitions.append)
    assert not transitions[-1].done and transitions[-1].reward == 0


def test_restoring_identical_visible_state_clears_each_seat_cycle_and_action_history():
    env = sampler()
    env.collect(5)  # Hero choice and four freezes: two closed loops.
    assert all(shaper.consecutive_cycles == 2 for shaper in env.opponent_shapers.values())
    state = env.batch.save(0)
    env.batch.load(0, state)
    env.collect(2)
    assert env.opponent_events == [[]]
    assert all(shaper.consecutive_cycles == 1 and shaper.turn_action_count == 2
               for shaper in env.opponent_shapers.values())
    env.collect(4)
    assert len(env.opponent_events[0]) == 2


@pytest.mark.parametrize("mode", ["cycle", "budget", "soft_cycle", "soft_forced_end"])
def test_role_aware_actor_and_synchronous_evaluation_match_over_two_rollouts(mode):
    costs = (CURRENT if mode == "cycle" else
             {**LEGACY, "turn_action_forfeit_limit": 4} if mode == "budget" else
             {**LEGACY, "cycle_cost": .01, "cycle_forfeit_repetitions": None,
              "turn_action_forfeit_limit": 60 if mode == "soft_cycle" else None})
    options = {"seed": 12, "threads": 1,
               "config": {"turn_action_caps": [60], "freeze_presses_per_turn": 0},
               "reward_shaping": costs, "frozen_opponents": 3}
    env = MixedOpponentVectorEnv(3, opponent_policy=pool(costs, old=1), **options)
    actor = UnifiedActorSampler(3, FreezeActor(), pool(costs, old=1), **options)
    observations = env.reset()
    expected = [[] for _ in range(3)]
    actual = [[] for _ in range(3)]
    try:
        length = 80 if mode.startswith("soft_") else 36
        for _ in range(length):
            actions = FreezeActor().act(observations).actions
            observations, rewards, dones, infos = env.step(actions)
            for game in range(3):
                expected[game].append((int(actions[game]), float(rewards[game]), bool(dones[game]),
                    infos[game].get("placement", 0),
                    {key: value[game].copy() for key, value in observations.items()}))

        def capture(transition):
            actual[transition.game].append((transition.action, float(transition.reward),
                                           transition.done,
                                           transition.placement, transition.next_observation))

        actor.collect(13, capture)
        actor.collect(length - 13, capture)
        for game in range(3):
            assert len(expected[game]) == len(actual[game]) == length
            for want, got in zip(expected[game], actual[game]):
                assert want[:4] == got[:4]
                for key, value in want[-1].items():
                    np.testing.assert_array_equal(value, got[-1][key])
            assert env.batch.save(game) == actor.batch.save(game)
    finally:
        env.close()


def prepared_game():
    cfg = {"hero_pool": ["TB_BaconShop_HERO_34"], "hero_choices": 1}
    game = tg.Game(11, json.dumps(cfg))
    while game.turn < 1:
        for seat in range(8):
            if game.needs_action(seat):
                game.step(seat, game.bot_action(seat))
    cards = [card["id"] for card in json.loads(tg.catalog_json())
             if card["kind"] == "minion" and card["in_pool"] and card["implemented"]][:2]
    for seat in (0, 1):
        for zone in ("board", "hand", "shop"):
            game.debug(json.dumps({"cmd": "clear_zone", "p": seat, "zone": zone}))
        game.debug(json.dumps({"cmd": "add_minion", "p": seat, "zone": "board",
                               "card": cards[seat]}))
    return game, cards


class ChosenActor(FreezeActor):
    def __init__(self, action):
        self.action = action
        self.seen = []

    def act(self, obs):
        self.seen.append({key: value.copy() for key, value in obs.items()})
        actions = np.full(len(obs["features"]), self.action, np.int64)
        assert obs["action_mask"][np.arange(len(actions)), actions].all()
        return ActorOutput(actions, np.zeros(len(actions), np.float32),
                           np.zeros(len(actions), np.float32))


@pytest.mark.parametrize("existing_discover", [False, True])
def test_deferred_opponent_action_does_not_advance_its_shaper(existing_discover):
    game, cards = prepared_game()
    game.debug(json.dumps({"cmd": "set_hero", "p": 0, "hero": "TB_BaconShop_HERO_25"}))
    game.debug(json.dumps({"cmd": "set_hero", "p": 1, "hero": "TB_BaconShop_HERO_70"}))
    game.debug(json.dumps({"cmd": "add_minion", "p": 1, "zone": "board", "card": cards[1]}))
    game.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "shop", "card": cards[0]}))
    game.debug(json.dumps({"cmd": "set_player", "p": 0, "health": 1, "armor": 0, "gold": 10}))
    game.step(0, 4)
    state = json.loads(game.state_json())
    if existing_discover:
        state["players"][1]["pending"] = [{"Discover": {
            "source": state["players"][1]["powers"][0]["card"],
            "options": [{"Minion": state["players"][1]["board"][0]}]}}]
    opponent = ChosenActor(30 if existing_discover else 203)
    fixed = FrozenOpponentPool((OpponentGroup("bot_v4", opponent, 1,
                                             {**CURRENT, "move_cost": .0005}),))
    env = UnifiedActorSampler(1, ChosenActor(30), fixed, frozen_opponents=1,
                              reward_shaping=LEGACY)
    episode = json.loads(env.batch.save(0))
    episode.update(game=state, seats=[0, 1])
    env.batch.load(0, json.dumps(episode))
    calls, snapshots = [], []
    original_shaping = env.opponent_shapers[1].step
    original_native = env.batch.step

    def shaped(*args, **kwargs):
        calls.append(args)
        return original_shaping(*args, **kwargs)

    def stepped(*args, **kwargs):
        result = original_native(*args, **kwargs)
        snapshots.append(json.loads(env.batch.save(0)))
        return result

    env.opponent_shapers[1].step = shaped
    env.batch.step = stepped
    transitions = []
    result = env.collect(1, transitions.append)
    assert result["deferred_opponent_actions"] == 1
    assert calls == [] and transitions[0].done and transitions[0].placement == 8
    assert env.opponent_shapers[1]._totals["move_actions"] == 0
    assert env.opponent_shapers[1]._totals["move_penalty"] == 0
    assert snapshots[0]["game"]["players"][1]["board"] == state["players"][1]["board"]
    assert len(snapshots[0]["game"]["players"][1]["pending"]) == 1 + int(existing_discover)


def test_budget_opponent_death_refreshes_bigglesworth_same_mask_discover():
    game, _ = prepared_game()
    game.debug(json.dumps({"cmd": "set_hero", "p": 0, "hero": "TB_BaconShop_HERO_70"}))
    state = json.loads(game.state_json())
    state["players"][0]["pending"] = [{"Discover": {
        "source": state["players"][0]["powers"][0]["card"],
        "options": [{"Minion": state["players"][0]["board"][0]}]}}]
    learner = ChosenActor(30)
    protocol = {**LEGACY, "turn_action_forfeit_limit": 1}
    env = UnifiedActorSampler(1, learner, pool(protocol, new=1), frozen_opponents=1,
                              reward_shaping=LEGACY)
    episode = json.loads(env.batch.save(0))
    episode.update(game=state, seats=[0, 1])
    env.batch.load(0, json.dumps(episode))
    env.sync_restored_history()
    shaper = env.opponent_shapers[1]
    shaper._action_turn = float(env.observations["features"][1, 0])
    shaper.turn_action_count = 1
    transitions = []
    env.collect(1, transitions.append)
    assert not transitions[0].done and transitions[0].action == 30
    assert env.batch.episode_index(0) == 0
    assert env.opponent_events[0][0]["reason"] == "action_limit_forfeit"
    assert transitions[0].next_observation["action_mask"][30]
    env.collect(1, transitions.append)
    assert len(learner.seen) == 2
    np.testing.assert_array_equal(learner.seen[0]["action_mask"], learner.seen[1]["action_mask"])
    assert not np.array_equal(learner.seen[0]["cards"], learner.seen[1]["cards"])
    assert transitions[1].action == 30 and not transitions[1].done
