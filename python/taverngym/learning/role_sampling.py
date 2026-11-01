"""Shared sampling when frozen opponents carry their own termination protocol.

Budget requests hold only the affected roles. Other sampled actions execute
before settlement, then all cycle/budget requests in that tick are eliminated
as one native batch. Learner transitions stay pending through every consequence.
"""

import time

import numpy as np

from .actor_sampling import ActorOutput, ActorTransition, _Pending
from .vector_env import NativeVectorEnv


def row(observations, slot):
    return {key: value[slot].copy() for key, value in observations.items() if key != "ready"}


class PredictActor:
    """Use the same frozen policy API in synchronous evaluation and sampling."""

    def __init__(self, policy):
        self.policy = policy
        self.requests = 0

    def act(self, obs):
        actions, _ = self.policy.predict(obs, deterministic=False)
        self.requests += len(actions)
        return ActorOutput(actions, np.zeros(len(actions), np.float32),
                           np.zeros(len(actions), np.float32))


class ExternalActionActor:
    def __init__(self, actions):
        self.actions = actions
        self.called = False

    def act(self, obs):
        if self.called or len(obs["features"]) != len(self.actions):
            raise RuntimeError("Synchronous step must submit its learner actions exactly once")
        self.called = True
        return ActorOutput(self.actions, np.zeros(len(self.actions), np.float32),
                           np.zeros(len(self.actions), np.float32))


def collect_role_aware(sampler, steps_per_game, on_transition=None, on_batch=None):
    if steps_per_game < 1:
        raise ValueError("steps_per_game must be positive")
    n, roles = sampler.n_games, sampler.roles
    counts = np.zeros(n, np.int64)
    pending = [None] * n
    slots = np.arange(sampler.batch.n_slots)
    submitted = native_steps = terminal_transitions = deferred = 0
    native_seconds = observation_seconds = reward_seconds = 0.0
    statistics = {}
    started = time.monotonic()

    def opponent_terminal(slot, reward, placement, reason, seats, completed_components=None):
        game, role = divmod(int(slot), roles)
        shaper = sampler.opponent_shapers.get(int(slot))
        metrics = completed_components
        if shaper is not None and metrics is None:
            if reason == "cycle_forfeit":
                metrics = shaper.finish_forfeit(reward, placement)
            else:
                metrics = shaper.finish_terminal(reward, placement, reason)
        identity = str(sampler.role_names[role])
        event = {"role": role, "policy": identity, "physical_seat": int(seats[slot]),
                 "placement": int(placement), "reason": reason or "native_terminal",
                 "reward_components": metrics}
        sampler.opponent_events[game].append(event)
        if reason:
            key = f"{identity}_{reason}s"
            statistics[key] = statistics.get(key, 0) + 1

    while (counts < steps_per_game).any():
        active = counts < steps_per_game
        full = sampler.observations
        ready = full["ready"] & active[slots // roles]
        seats = sampler.batch.slot_seats.copy()
        budgets = np.zeros(len(slots), bool)
        for slot, shaper in sampler.opponent_shapers.items():
            if ready[slot]:
                budgets[slot] = shaper.action_limit_exceeded(row(full, slot))
        for game, shaper in enumerate(sampler.shapers):
            slot = game * roles
            if ready[slot]:
                budgets[slot] = shaper.action_limit_exceeded(row(full, slot))
        actions = np.zeros(len(slots), np.int64)
        for identity, actor in sampler.actors.items():
            selected = np.flatnonzero(ready & (sampler.role_names[slots % roles] == identity)
                                     & ((slots % roles == 0) | ~budgets))
            if not len(selected):
                continue
            observations = {key: value[selected] for key, value in full.items() if key != "ready"}
            output = actor.act(observations)
            actions[selected] = output.actions
            if identity == "learner":
                for index, slot in enumerate(selected):
                    game = int(slot // roles)
                    if pending[game] is not None:
                        raise RuntimeError("Learner has an unfinished action")
                    pending[game] = _Pending(row(full, slot), int(output.actions[index]),
                        np.float32(output.values[index]), np.float32(output.log_probs[index]))
                    submitted += 1
        if any(pending[game] is None for game in np.flatnonzero(active)):
            raise RuntimeError("Active game has no real pending learner request")
        tick = time.monotonic()
        rewards, dones, metadata = sampler.batch.step(actions, active=active, skipped=budgets)
        NativeVectorEnv._validate_result(dones, metadata)
        if metadata["deferred"][::roles].any():
            raise RuntimeError("First-role learner action unexpectedly deferred")
        deferred += int(metadata["deferred"].sum())
        native_seconds += time.monotonic() - tick
        tick = time.monotonic()
        after = sampler.batch.observe()
        observation_seconds += time.monotonic() - tick
        tick = time.monotonic()
        # A skipped budget request is not a normal action. Deferred actions also
        # leave that seat's action and cycle histories untouched.
        requested = budgets.copy()
        reasons = {int(slot): "action_limit_forfeit" for slot in np.flatnonzero(budgets)}
        for slot in np.flatnonzero(dones):
            requested[slot] = False
            reasons.pop(int(slot), None)
        for slot, shaper in sampler.opponent_shapers.items():
            if dones[slot]:
                continue
            if ready[slot] and not budgets[slot] and not metadata["deferred"][slot]:
                shaper.step(actions[slot], row(full, slot), row(after, slot), float(rewards[slot]))
                if shaper.forfeit_requested:
                    requested[slot] = True
                    reasons[slot] = "cycle_forfeit"
        for slot in np.flatnonzero(dones & (slots % roles != 0)):
            components = None
            shaper = sampler.opponent_shapers.get(int(slot))
            if (shaper is not None and ready[slot] and not budgets[slot]
                    and not metadata["deferred"][slot]):
                _, components = shaper.step(actions[slot], row(full, slot), row(after, slot),
                                             float(rewards[slot]), done=True)
            opponent_terminal(slot, float(rewards[slot]), int(metadata["placement"][slot]),
                              None, seats, components)
        for game in np.flatnonzero(active):
            pending[game].reward += float(rewards[game * roles])
        endings = {}
        # Processing elimination may make a previously waiting learner ready or
        # terminal. Repeat without selecting or executing another policy action.
        for _ in range(roles + 2):
            for game in np.flatnonzero(active):
                slot = int(game * roles)
                if game in endings or (not dones[slot] and not after["ready"][slot]):
                    continue
                if budgets[slot] and not dones[slot]:
                    continue  # Its intercepted request is shaped after native elimination.
                current = pending[game]
                done = bool(dones[slot])
                placement = int(metadata["placement"][slot]) if done else 0
                reason = reasons.get(slot) if done else None
                reward, components = current.reward, None
                if sampler.shapers:
                    reward, components = sampler.shapers[game].step(
                        current.action, current.observation, row(after, slot), reward,
                        done=done, terminal_reason=reason)
                    if sampler.shapers[game].forfeit_requested:
                        requested[slot] = True
                        reasons[slot] = "cycle_forfeit"
                endings[game] = {"reward": reward, "components": components,
                                 "done": done, "placement": placement, "reason": reason}
            selected = requested & ~dones
            if not selected.any():
                break
            # Earlier real actions can naturally eliminate a requested seat; no
            # second death/reward is synthesized for it or for an auto-reset game.
            if not after["ready"][selected].all():
                raise RuntimeError("Termination request lost its ready native seat")
            terminal_rewards, terminal_dones, terminal_info = sampler.batch.forfeit(selected)
            NativeVectorEnv._validate_result(terminal_dones, terminal_info)
            if not terminal_dones[selected].all():
                raise RuntimeError("Native simultaneous surrender left requested roles alive")
            requested[selected] = False
            for slot in np.flatnonzero(terminal_dones & (slots % roles != 0)):
                opponent_terminal(slot, float(terminal_rewards[slot]),
                    int(terminal_info["placement"][slot]), reasons.get(int(slot)), seats)
            for game in np.flatnonzero(active):
                slot = int(game * roles)
                delta = float(terminal_rewards[slot])
                pending[game].reward += delta
                if game in endings:
                    ending = endings[game]
                    ending["reward"] += delta
                    if terminal_dones[slot]:
                        reason = reasons.get(slot)
                        placement = int(terminal_info["placement"][slot])
                        if sampler.shapers:
                            ending["components"] = (
                                sampler.shapers[game].finish_forfeit(delta, placement)
                                if reason == "cycle_forfeit" else
                                sampler.shapers[game].finish_terminal(delta, placement))
                        ending.update(done=True, placement=placement, reason=reason)
            metadata["placement"][terminal_dones] = terminal_info["placement"][terminal_dones]
            dones |= terminal_dones
            after = sampler.batch.observe()
        else:
            raise RuntimeError("Termination settlement exceeded its finite role budget")
        # Reset only the learner's finished game, after all role results were captured.
        sampler.batch.reset_finished_first()
        after = sampler.batch.observe()
        completed = []
        for game, ending in endings.items():
            slot = game * roles
            current = pending[game]
            components = ending["components"]
            if components is not None:
                components["opponent_cycle_forfeits"] = sum(
                    event["reason"] == "cycle_forfeit" for event in sampler.opponent_events[game])
                components["opponent_action_limit_forfeits"] = sum(
                    event["reason"] == "action_limit_forfeit"
                    for event in sampler.opponent_events[game])
            reward, info, done, placement = sampler._episode_info(game, current,
                ending["reward"], ending["done"], ending["placement"], components, ending["reason"])
            if done:
                info["opponent_terminations"] = sampler.opponent_events[game]
                sampler.opponent_events[game] = []
                for opponent_slot, shaper in sampler.opponent_shapers.items():
                    if opponent_slot // roles == game:
                        shaper.reset()
            transition = ActorTransition(int(game), int(sampler.episodes[game]),
                int(sampler.episode_steps[game]), int(counts[game]), current.observation,
                current.action, current.value, current.log_prob,
                np.float32(reward), done, placement,
                row(after, slot), info)
            if on_transition:
                on_transition(transition)
            if on_batch:
                completed.append(transition)
            counts[game] += 1
            sampler.episode_steps[game] += 1
            if done:
                terminal_transitions += 1
                sampler.episodes[game] += 1
                sampler.episode_steps[game] = 0
            pending[game] = None
        if completed:
            on_batch(completed)
        sampler.observations = after
        reward_seconds += time.monotonic() - tick
        native_steps += 1
        if native_steps > max(10000, steps_per_game * 200):
            raise RuntimeError("Role-aware collection exceeded its native-step budget")
    assert submitted == int(counts.sum()) == n * steps_per_game
    assert all(value is None for value in pending)
    assert sampler.observations["ready"][::roles].all()
    seconds = time.monotonic() - started
    return {"learner_samples": submitted, "sampling_seconds": seconds,
            "learner_samples_per_second": submitted / seconds, "native_steps": native_steps,
            "terminal_transitions": terminal_transitions, "native_seconds": native_seconds,
            "deferred_opponent_actions": deferred, "observation_seconds": observation_seconds,
            "reward_and_sink_seconds": reward_seconds, **statistics}
