"""Training rewards for action cycles; observations and legality stay native.

Only public observation arrays enter cycle detection. History is reward bookkeeping,
not an inference filter or an extra model input. A cycle is a return to an identical
visible state. Legacy configs retain the original Move/Freeze segment detector;
new configs can opt into checking every action within a recruit turn.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import math

import taverngym as tg

from .model import ObservationLayout


REWARD_METRICS = (
    "placement_reward", "action_reward", "operation_penalty", "cycle_penalty", "total_reward",
    "reversible_actions", "cycle_returns", "actions_per_game",
    "cycle_forfeit_rate", "cycle_forfeit_turn", "cycle_forfeit_placement",
    "max_consecutive_cycles",
    "action_limit_forfeit_rate", "action_limit_forfeit_turn", "action_limit_forfeit_placement",
    "opponent_cycle_forfeits", "opponent_action_limit_forfeits",
    "cycle_return_fraction", "max_consecutive_cycle_returns",
    "move_actions", "move_penalty", "move_action_fraction",
    "early_actions", "early_chargeable_actions", "early_menu_actions", "early_action_penalty",
    *(f"early_turn{turn}_actions" for turn in range(1, 6)),
    *(f"early_turn{turn}_observed" for turn in range(1, 6)),
    "first_turn_minion_bonus", "first_turn_minion_bonus_count",
    "second_turn_tier2_bonus", "second_turn_tier2_bonus_count",
    "third_turn_dark_gift_bonus", "third_turn_dark_gift_bonus_count", "opening_bonus",
    "hero_selection_bonus", "hero_selection_bonus_count", "hero_selection_count",
    "hero_selection_scorable", "hero_selection_unknown_offer", "hero_selection_known_fraction",
    "hero_selection_offered_heroes", "hero_selection_unknown_heroes",
)

OPENING_BONUS_FIELDS = (
    "first_turn_minion_bonus", "second_turn_tier2_bonus", "third_turn_dark_gift_bonus",
)

ACTION_REWARD_COMPONENTS = (
    "operation_penalty", "cycle_penalty", "move_penalty", "early_action_penalty", "opening_bonus",
)


@lru_cache(maxsize=1)
def _active_hero_indices():
    """Resolve stable IDs once per process; observations use CardId + 1."""
    return {card["id"]: card["cid"] + 1 for card in json.loads(tg.catalog_json())
            if card["kind"] == "hero" and card["in_pool"] and card["implemented"]}


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
    # Per successful non-End recruit action in turns 1..5; old archives stay off.
    early_action_costs: tuple[float, ...] = (0., 0., 0., 0., 0.)
    early_action_penalty_cap: float = .01
    first_turn_minion_bonus: float = 0.
    second_turn_tier2_bonus: float = 0.
    third_turn_dark_gift_bonus: float = 0.
    hero_selection_bonus_scale: float = 0.
    hero_selection_bonus_cap: float = .005
    hero_selection_mean_placements: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0
               for value in (self.operation_cost, self.cycle_cost, self.move_cost,
                             self.early_action_penalty_cap,
                             self.hero_selection_bonus_scale, self.hero_selection_bonus_cap,
                             *(getattr(self, name) for name in OPENING_BONUS_FIELDS))):
            raise ValueError("Reward costs must be finite and nonnegative")
        costs = self.early_action_costs
        if (not isinstance(costs, (list, tuple)) or len(costs) != 5
                or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                       for value in costs)):
            raise ValueError("early_action_costs must contain five finite nonnegative costs")
        object.__setattr__(self, "early_action_costs", tuple(costs))
        means = self.hero_selection_mean_placements
        if (not isinstance(means, dict)
                or any(not isinstance(key, str) or type(value) not in (int, float)
                       or not math.isfinite(value) or not 1 <= value <= 8
                       for key, value in means.items())):
            raise ValueError(
                "Hero selection means must map stable hero IDs to finite placements 1..8")
        if self.hero_selection_bonus_scale > 0:
            active = _active_hero_indices()
            if any(key not in active for key in means):
                raise ValueError("Hero selection prior contains an unknown or non-active hero ID")
        object.__setattr__(self, "hero_selection_mean_placements", dict(means))
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
    """Normalize only optional move/opening defaults in old archives.

    Keep every other field strict, including None versus an enabled protocol.
    Do not mutate the dictionaries used for source hashes and historical records.
    """
    if left is None or right is None:
        return left is right
    def normalized(protocol):
        defaults = {"move_cost": 0, "early_action_costs": (0.,) * 5,
                    "early_action_penalty_cap": .01,
                    "hero_selection_bonus_scale": 0., "hero_selection_bonus_cap": .005,
                    "hero_selection_mean_placements": {},
                    **dict.fromkeys(OPENING_BONUS_FIELDS, 0.)}
        result = {**defaults, **protocol}
        if isinstance(result["early_action_costs"], (list, tuple)):
            result["early_action_costs"] = tuple(result["early_action_costs"])
        return result

    return normalized(left) == normalized(right)


class ActionRewardShaper:
    """One instance per learner seat; reset at environment/episode boundaries."""

    def __init__(self, config: RewardShapingConfig):
        self.config = config
        self.layout = ObservationLayout()
        self._hero_means = ({_active_hero_indices()[key]: value
                             for key, value in config.hero_selection_mean_placements.items()}
                            if config.hero_selection_bonus_scale > 0 else {})
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
        self._early_turn = None
        self._early_penalty_spent = 0.
        self._early_chargeable_count = 0
        self._early_observed_turns = set()
        self._opening_awarded = set()
        self._dark_gift_transaction = None
        self._hero_selection_completed = False

    def _hero_selection_step(self, action, before, after, *, done, executed):
        if self.config.hero_selection_bonus_scale <= 0 or self._hero_selection_completed:
            return {}
        native = self.layout.native
        pending = native["GLOBAL_F"] + native["N_QUESTS"] * native["QUEST_F"]
        index = action - native["A_CHOOSE"]
        if (not executed or done or before["features"][0] != 0
                or not before["features"][1] or not before["features"][pending + 1]
                or not 0 <= index < native["N_CHOICE"] or not before["action_mask"][action]
                or after["features"][2] or not after["features"][4]):
            return {}
        offered = before["cards"][self.layout.cards["choices"]]
        selected = int(offered[index])
        if selected <= 0 or after["cards"][0] != selected or after["features"][pending + 1]:
            return {}  # No completed HeroSelect, or a terminal/reset/menu observation.
        actual = {int(hero) for i, hero in enumerate(offered)
                  if hero > 0 and before["action_mask"][native["A_CHOOSE"] + i]}
        unknown = actual - self._hero_means.keys()
        scorable = len(actual) >= 2 and not unknown
        self._hero_selection_completed = True
        bonus = 0.
        if scorable:
            difference = max(self._hero_means[hero] for hero in actual) - self._hero_means[selected]
            bonus = min(self.config.hero_selection_bonus_cap,
                        self.config.hero_selection_bonus_scale * max(0., difference))
        return {"hero_selection_bonus": bonus, "hero_selection_bonus_count": int(bonus > 0),
                "hero_selection_count": 1, "hero_selection_scorable": int(scorable),
                "hero_selection_unknown_offer": int(bool(unknown)),
                "hero_selection_known_fraction": (len(actual) - len(unknown)) / len(actual),
                "hero_selection_offered_heroes": len(actual),
                "hero_selection_unknown_heroes": len(unknown)}

    def _opening_step(self, action, before, after, *, done, executed):
        """Visible-state milestones and a separate per-turn cost ledger.

        Cycle-history resets and rollout boundaries must not reset this ledger.
        The caller submits executed actions only, except a marked budget forfeit.
        """
        values = {}
        turn = int(round(float(before["features"][0]) * 20))
        after_turn = int(round(float(after["features"][0]) * 20))
        native = self.layout.native
        end = action == native["A_END"]
        if turn != self._early_turn:
            self._early_turn = turn
            self._early_penalty_spent = 0.
            self._early_chargeable_count = 0
            self._dark_gift_transaction = None
        if executed and 1 <= turn <= 5:
            values["early_actions"] = 1
            values[f"early_turn{turn}_actions"] = 1
            if turn not in self._early_observed_turns:
                self._early_observed_turns.add(turn)
                values[f"early_turn{turn}_observed"] = 1
            values["early_menu_actions"] = int(
                action == native["A_CANCEL"] or native["A_CHOOSE"] <= action < native["A_PLAY"])
            if not end:
                values["early_chargeable_actions"] = 1
                self._early_chargeable_count += 1
                target = min(
                    self._early_chargeable_count * self.config.early_action_costs[turn - 1],
                    self.config.early_action_penalty_cap)
                charge = max(0., target - self._early_penalty_spent)
                self._early_penalty_spent = target
                values["early_action_penalty"] = -charge

        # Auto-reset observations can resemble an opening state. Never infer
        # milestones from terminal, End, hero-selection or cross-turn output.
        same_recruit = (executed and not end and not done and turn == after_turn
                        and not before["features"][1] and not after["features"][1])
        if same_recruit:
            span, count, width = self.layout.floats["board"]
            old_board = before["features"][span].reshape(count, width)[:, 0].any()
            new_board = after["features"][span].reshape(count, width)[:, 0].any()
            if turn == 1 and not old_board and new_board:
                self._award_opening("first_turn_minion_bonus", values)
            old_tier = int(round(float(before["features"][10]) * 6))
            new_tier = int(round(float(after["features"][10]) * 6))
            if turn == 2 and action == native["A_UPGRADE"] and (old_tier, new_tier) == (1, 2):
                self._award_opening("second_turn_tier2_bonus", values)
            pending_offset = native["GLOBAL_F"] + native["N_QUESTS"] * native["QUEST_F"]
            source_index = 1 + native["N_POWERS"] + 1  # hero, powers, Deity, pending source

            def discover_identity(obs):
                if not obs["features"][pending_offset + 2]:
                    return None
                return (int(round(float(obs["features"][23]) * 3)),
                        int(obs["cards"][source_index]))

            if turn == 3:
                prior = self._dark_gift_transaction
                if prior is not None:
                    if (discover_identity(before) == prior and
                            native["A_CHOOSE"] <= action < native["A_PLAY"]):
                        self._award_opening("third_turn_dark_gift_bonus", values)
                    # Only the immediately matching native Discover can complete
                    # this transaction. Cancel or another menu cannot reuse it.
                    self._dark_gift_transaction = None
                if action == native["A_DARK_GIFT"]:
                    old_uses = int(round(float(before["features"][23]) * 3))
                    identity = discover_identity(after)
                    if identity is not None and identity[0] == old_uses + 1 and identity[1] > 0:
                        self._dark_gift_transaction = identity
        elif self._dark_gift_transaction is not None:
            self._dark_gift_transaction = None
        if end or after_turn != turn or done:
            self._early_turn = None
            self._early_penalty_spent = 0.
            self._early_chargeable_count = 0
            self._dark_gift_transaction = None
        return values

    def _award_opening(self, name, values):
        amount = getattr(self.config, name)
        if amount and name not in self._opening_awarded:
            self._opening_awarded.add(name)
            values[name] = amount
            values[f"{name}_count"] = 1
            values["opening_bonus"] = values.get("opening_bonus", 0.) + amount

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

    def _action_step(self, action, before, after, native_reward, *, done, terminal_reason):
        """Compute all action costs/bonuses once, retaining their diagnostic fields.

        The first result includes native reward in the historical addition order
        so grouping does not change the actual float32 rewards/GAE at rounding
        boundaries. action_reward is a derived subtotal, never an extra payment.
        """
        opening = self._opening_step(action, before, after, done=done,
                                     executed=terminal_reason != "action_limit_forfeit")
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
        values = {"operation_penalty": operation_penalty, "cycle_penalty": cycle_penalty,
                  "reversible_actions": int(reversible),
                  "cycle_returns": int(cycle), "actions_per_game": 1,
                  "move_actions": int(move), "move_penalty": move_penalty, **opening}
        components = [values.get(name, 0.) for name in ACTION_REWARD_COMPONENTS]
        values["action_reward"] = sum(components)
        reward = float(native_reward)
        for component in components:
            reward += component
        return reward, values

    def step(self, action, before, after, native_reward, *, done=False, terminal_reason=None):
        """Return one scalar reward and optional completed-episode diagnostics.

        Decomposition: placement_reward + hero_selection_bonus + action_reward.
        `after` can be auto-reset when done=True. Call only for executed actions,
        except the explicitly marked intercepted budget request. Deferred and
        rejected actions must not enter this method.
        """
        action = int(action)
        hero = self._hero_selection_step(action, before, after, done=done,
                                         executed=terminal_reason != "action_limit_forfeit")
        reward, operations = self._action_step(action, before, after, native_reward,
                                               done=done, terminal_reason=terminal_reason)
        reward += hero.get("hero_selection_bonus", 0.)
        values = {"placement_reward": float(native_reward), "total_reward": reward,
                  **operations, **hero}
        for name, value in values.items():
            self._totals[name] += value
        if done:
            if terminal_reason == "action_limit_forfeit":
                self._totals["action_limit_forfeit_rate"] = 1
                self._totals["action_limit_forfeit_turn"] = round(float(before["features"][0]) * 20)
            metrics = self._completed_metrics()
            self.reset()
            return reward, metrics
        return reward, None
