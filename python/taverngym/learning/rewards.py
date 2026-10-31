"""Training rewards for reversible actions; observations and legality stay native.

Only public observation arrays enter cycle detection. History is reward bookkeeping,
not an inference filter or an extra model input. A cycle is a return to an identical
visible state within an uninterrupted Move/Freeze segment with no other progress.
"""

from dataclasses import dataclass
import hashlib
import math

from .model import ObservationLayout


REWARD_METRICS = (
    "placement_reward", "operation_penalty", "cycle_penalty", "total_reward",
    "reversible_actions", "cycle_returns", "actions_per_game",
)


@dataclass(frozen=True)
class RewardShapingConfig:
    # Positive magnitudes: these amounts are subtracted from native reward.
    operation_cost: float = 0.0001
    cycle_cost: float = 0.0002

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0
               for value in (self.operation_cost, self.cycle_cost)):
            raise ValueError("Reward costs must be finite and nonnegative")


class ActionRewardShaper:
    """One instance per learner seat; reset at environment/episode boundaries."""

    def __init__(self, config: RewardShapingConfig):
        self.config = config
        self.layout = ObservationLayout()
        self.reset()

    def _clear_history(self):
        self._context = None
        self._seen = set()

    def reset(self):
        self._clear_history()
        self._totals = dict.fromkeys(REWARD_METRICS, 0.0)

    def _state(self, obs):
        features, cards = obs["features"], obs["cards"]
        if features.ndim != 1 or cards.ndim != 1:
            raise ValueError("Reward history requires one unbatched player observation")
        # Ignore the legal mask: action counters can change legality while the
        # board/shop still repeats. Keep full visible state for the cycle key.
        visible = hashlib.sha256(features.tobytes() + cards.tobytes()).digest()
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

    def step(self, action, before, after, native_reward, *, done=False):
        """Return (training reward, completed-episode metrics or None).

        `after` may already be an automatically reset game when done=True. Never
        compare that observation with the previous episode's history.
        """
        action = int(action)
        move_start = self.layout.native["A_MOVE"]
        move_stop = move_start + self.layout.native["N_BOARD"] ** 2
        reversible = action == 2 or move_start <= action < move_stop
        cycle = False
        if reversible and not done:
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
            # Purchases, plays, refreshes, choices and EndTurn start a fresh
            # decision segment even if their visible result happens to match.
            self._clear_history()
        operation_penalty = -self.config.operation_cost if reversible else 0.0
        cycle_penalty = -self.config.cycle_cost if cycle else 0.0
        reward = float(native_reward) + operation_penalty + cycle_penalty
        values = (float(native_reward), operation_penalty, cycle_penalty, reward,
                  int(reversible), int(cycle), 1)
        for name, value in zip(REWARD_METRICS, values):
            self._totals[name] += value
        if done:
            metrics = self._totals.copy()
            self.reset()
            return reward, metrics
        return reward, None
