"""Training rewards for action cycles; observations and legality stay native.

Only public observation arrays enter cycle detection. History is reward bookkeeping,
not an inference filter or an extra model input. A cycle is a return to an identical
visible state. Legacy configs retain the original Move/Freeze segment detector;
new configs can opt into checking every action within a recruit turn.
"""

from dataclasses import dataclass
import hashlib
import math

from .model import ObservationLayout


REWARD_METRICS = (
    "placement_reward", "operation_penalty", "cycle_penalty", "total_reward",
    "reversible_actions", "cycle_returns", "actions_per_game",
    "cycle_forfeit_rate", "cycle_forfeit_turn", "cycle_forfeit_placement",
    "max_consecutive_cycles",
    "action_limit_forfeit_rate", "action_limit_forfeit_turn", "action_limit_forfeit_placement",
    "opponent_cycle_forfeits", "opponent_action_limit_forfeits",
    "cycle_return_fraction", "max_consecutive_cycle_returns",
    "move_actions", "move_penalty", "move_action_fraction",
)


@dataclass(frozen=True)
class RewardShapingConfig:
    # Positive magnitudes: these amounts are subtracted from native reward.
    operation_cost: float = 0.0001
    cycle_cost: float = 0.0002
    # Missing in historical experiment configs: preserve their reward semantics.
    cycle_detection: str = "reversible_segment"
    # Explicit V4+ experiment opt-in; never inferred from a model filename.
    cycle_forfeit_repetitions: int | None = None
    turn_action_forfeit_limit: int | None = None
    # Separate from the historical combined Move/Freeze operation cost.
    # Append with a zero default to preserve old configs and positional callers.
    move_cost: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0
               for value in (self.operation_cost, self.cycle_cost, self.move_cost)):
            raise ValueError("Reward costs must be finite and nonnegative")
        if self.cycle_detection not in ("reversible_segment", "all_actions"):
            raise ValueError("cycle_detection must be 'reversible_segment' or 'all_actions'")
        count = self.cycle_forfeit_repetitions
        if count is not None and (type(count) is not int or count < 1):
            raise ValueError("cycle_forfeit_repetitions must be a positive integer or null")
        if count is not None and self.cycle_detection != "all_actions":
            raise ValueError("Cycle forfeits require cycle_detection='all_actions'")
        limit = self.turn_action_forfeit_limit
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("turn_action_forfeit_limit must be a positive integer or null")


def reward_protocols_equal(left, right):
    """Only a missing move_cost is equivalent to explicit zero in old archives.

    Keep every other field strict, including None versus an enabled protocol.
    Do not mutate the dictionaries used for source hashes and historical records.
    """
    if left is None or right is None:
        return left is right
    return {**left, "move_cost": left.get("move_cost", 0)} == {
        **right, "move_cost": right.get("move_cost", 0)}


class ActionRewardShaper:
    """One instance per learner seat; reset at environment/episode boundaries."""

    def __init__(self, config: RewardShapingConfig):
        self.config = config
        self.layout = ObservationLayout()
        self.reset()

    def _clear_history(self):
        self._context = None
        self._seen = set()
        self._turn = None
        self._last_visible = None
        self._path = []
        self._path_index = {}
        self.consecutive_cycles = 0
        self.consecutive_returns = 0
        self.forfeit_requested = False

    def reset(self):
        self._clear_history()
        self._totals = dict.fromkeys(REWARD_METRICS, 0.0)
        self._action_turn = None
        self.turn_action_count = 0

    def _completed_metrics(self):
        # Compute per game before aggregation: mean(R/A) is not mean(R)/mean(A).
        self._totals["cycle_return_fraction"] = (
            self._totals["cycle_returns"] / max(self._totals["actions_per_game"], 1))
        # Denominator matches cycle_return_fraction: all real policy transitions
        # in this episode, including hero selection; only recruit Moves count.
        self._totals["move_action_fraction"] = (
            self._totals["move_actions"] / max(self._totals["actions_per_game"], 1))
        return self._totals.copy()

    def action_limit_exceeded(self, before):
        """True when the NEXT action would exceed this recruit turn's budget.

        Intercept before native End can advance combat or reset the game. The
        attempted over-budget action is the terminal policy transition.
        """
        limit = self.config.turn_action_forfeit_limit
        turn = float(before["features"][0])
        return (limit is not None and turn > 0 and turn == self._action_turn
                and self.turn_action_count >= limit)

    @staticmethod
    def _visible(obs):
        features, cards = obs["features"], obs["cards"]
        if features.ndim != 1 or cards.ndim != 1:
            raise ValueError("Reward history requires one unbatched player observation")
        # Ignore the legal mask: action counters can change legality while the
        # board/shop still repeats. Keep full visible state for the cycle key.
        return hashlib.sha256(features.tobytes() + cards.tobytes()).digest()

    def _state(self, obs):
        visible = self._visible(obs)
        features, cards = obs["features"], obs["cards"]
        context_f, context_c = features.copy(), cards.copy()
        span, count, width = self.layout.floats["board"]
        board = features[span].reshape(count, width)
        board_cards = cards[self.layout.cards["board"]]
        order = sorted(range(count), key=lambda i: (int(board_cards[i]), board[i].tobytes()))
        context_f[span] = board[order].reshape(-1)
        context_c[self.layout.cards["board"]] = board_cards[order]
        # obs.rs: global[15] = any shop frozen; item header[6] = frozen.
        context_f[15] = 0
        shop_span, shop_count, shop_width = self.layout.floats["shop"]
        context_f[shop_span].reshape(shop_count, shop_width)[:, 6] = 0
        context = hashlib.sha256(context_f.tobytes() + context_c.tobytes()).digest()
        return context, visible

    def _all_actions_cycle(self, before, after):
        old_visible, visible = self._visible(before), self._visible(after)
        turn = float(before["features"][0])
        if turn != float(after["features"][0]):
            self._clear_history()
            return False
        if self._turn != turn or self._last_visible != old_visible:
            # A restored state or intervening public changes break continuity.
            self._clear_history()
        self._turn = turn
        self._seen.add(old_visible)
        cycle = visible in self._seen
        if self.config.cycle_forfeit_repetitions is not None:
            self._count_complete_cycles(old_visible, visible)
        self._seen.add(visible)
        self._last_visible = visible
        return cycle

    def _count_complete_cycles(self, before, after):
        # Erase each closed path after counting it. A->B->A is ONE loop;
        # re-entering B alone is not another. This also counts longer menu
        # cycles and changing between already-visited paths without progress.
        if not self._path:
            self._path = [before]
            self._path_index = {before: 0}
        if after not in self._seen:
            self.consecutive_cycles = 0
        if after in self._path_index:
            self.consecutive_cycles += 1
            cut = self._path_index[after] + 1
            for state in self._path[cut:]:
                del self._path_index[state]
            del self._path[cut:]
        else:
            self._path_index[after] = len(self._path)
            self._path.append(after)
        self._totals["max_consecutive_cycles"] = max(
            self._totals["max_consecutive_cycles"], self.consecutive_cycles)
        self.forfeit_requested = self.consecutive_cycles >= self.config.cycle_forfeit_repetitions

    def finish_forfeit(self, native_reward, placement):
        """Finish after the native environment has actually eliminated the role.

        step() has already charged this action's cycle penalty. Add exactly one
        native placement reward without adding a fabricated learner action.
        """
        if not self.forfeit_requested or not 1 <= placement <= 8:
            raise ValueError("Cycle forfeit needs a request and a native final placement")
        self._totals["placement_reward"] += float(native_reward)
        self._totals["total_reward"] += float(native_reward)
        self._totals["cycle_forfeit_rate"] = 1
        self._totals["cycle_forfeit_turn"] = round(self._turn * 20)
        self._totals["cycle_forfeit_placement"] = placement
        metrics = self._completed_metrics()
        self.reset()
        return metrics

    def finish_terminal(self, native_reward, placement, reason=None):
        """Finish without inventing an executed action (e.g. opponent-only deaths).

        Opponent action-budget termination records the intercepted next request;
        a normal learner victory created by those deaths adds no learner action.
        """
        if not 1 <= placement <= 8:
            raise ValueError("Terminal metrics require a native final placement")
        self._totals["placement_reward"] += float(native_reward)
        self._totals["total_reward"] += float(native_reward)
        if reason == "action_limit_forfeit":
            self._totals["actions_per_game"] += 1
            self._totals["action_limit_forfeit_rate"] = 1
            self._totals["action_limit_forfeit_turn"] = round(self._action_turn * 20)
            self._totals["action_limit_forfeit_placement"] = placement
        metrics = self._completed_metrics()
        self.reset()
        return metrics

    def step(self, action, before, after, native_reward, *, done=False, terminal_reason=None):
        """Return (training reward, completed-episode metrics or None).

        `after` may already be an automatically reset game when done=True. Never
        compare that observation with the previous episode's history.
        Call only for executed actions, except the explicit intercepted budget
        request below. Deferred/skipped actions must not enter this method.
        """
        action = int(action)
        turn = float(before["features"][0])
        if turn != self._action_turn:
            self.turn_action_count = 0
            self._action_turn = turn
        self.turn_action_count += int(turn > 0)
        move_start = self.layout.native["A_MOVE"]
        move_stop = move_start + self.layout.native["N_BOARD"] ** 2
        move = turn > 0 and move_start <= action < move_stop
        reversible = action == 2 or move_start <= action < move_stop
        if terminal_reason == "action_limit_forfeit":
            # This over-budget request was intercepted, not executed.
            reversible = False
            move = False
        cycle = False
        if done or action == self.layout.native["A_END"]:
            # End and terminal/reset observations never receive a cycle cost.
            self._clear_history()
        elif self.config.cycle_detection == "all_actions":
            cycle = self._all_actions_cycle(before, after)
        elif reversible:
            old_context, old_visible = self._state(before)
            context, visible = self._state(after)
            if old_context != self._context:
                self._clear_history()
                self._context = old_context
            self._seen.add(old_visible)
            if context == old_context:
                cycle = visible in self._seen
            else:
                # Stats, resources, choices, hand, turn or any other visible
                # change can justify revisiting a board order / Freeze state.
                self._clear_history()
                self._context = context
            self._seen.add(visible)
        else:
            # Preserve historical non-reversible-action reset behavior.
            self._clear_history()
        operation_penalty = -self.config.operation_cost if reversible else 0.0
        move_penalty = -self.config.move_cost if move else 0.0
        cycle_penalty = -self.config.cycle_cost if cycle else 0.0
        self.consecutive_returns = self.consecutive_returns + 1 if cycle else 0
        self._totals["max_consecutive_cycle_returns"] = max(
            self._totals["max_consecutive_cycle_returns"], self.consecutive_returns)
        reward = float(native_reward) + operation_penalty + cycle_penalty + move_penalty
        values = {"placement_reward": float(native_reward),
                  "operation_penalty": operation_penalty, "cycle_penalty": cycle_penalty,
                  "total_reward": reward, "reversible_actions": int(reversible),
                  "cycle_returns": int(cycle), "actions_per_game": 1,
                  "move_actions": int(move), "move_penalty": move_penalty}
        for name, value in values.items():
            self._totals[name] += value
        if done:
            if terminal_reason == "action_limit_forfeit":
                self._totals["action_limit_forfeit_rate"] = 1
                self._totals["action_limit_forfeit_turn"] = round(turn * 20)
            metrics = self._completed_metrics()
            self.reset()
            return reward, metrics
        return reward, None
