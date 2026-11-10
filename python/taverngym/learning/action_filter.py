"""Opt-in history filter for reversible action loops, separate from native legality.

Use one instance per player session. Call mask() before a recommendation and
commit() only after the action really executes. This never changes model weights.
"""

import hashlib

import numpy as np

from .model import ObservationLayout


class ReversibleActionFilter:
    def __init__(self, max_reversible_actions=8):
        if max_reversible_actions < 1:
            raise ValueError("max_reversible_actions must be positive")
        self.max_reversible_actions = max_reversible_actions
        self.layout = ObservationLayout()
        self.reset()

    def reset(self):
        self._context = None
        self._boards = set()
        self._freezes = set()
        self.reversible_actions = 0
        self.last_blocked = {}

    def _state(self, obs):
        f, c = obs["features"], obs["cards"]
        if f.ndim != 1 or c.ndim != 1:
            raise ValueError("Use one filter per seat and pass one unbatched observation")
        span, count, width = self.layout.floats["board"]
        board = f[span].reshape(count, width)
        board_cards = c[self.layout.cards["board"]]
        rows = tuple((int(card), row.tobytes()) for card, row in zip(board_cards, board))
        shop_span, shop_count, shop_width = self.layout.floats["shop"]
        shop = f[shop_span].reshape(shop_count, shop_width)
        present = shop[:, 0] > 0
        frozen = tuple(bool(value) for value in shop[:, 6])
        # Normalize only order and Freeze flags. Every other visible change resets
        # history, including prices, gold, hand, stats, offers, choices and turn.
        canonical = sorted(range(count), key=lambda i: rows[i])
        context_f, context_c = f.copy(), c.copy()
        context_f[span] = board[canonical].reshape(-1)
        context_c[self.layout.cards["board"]] = board_cards[canonical]
        context_f[15] = 0
        context_f[shop_span].reshape(shop_count, shop_width)[:, 6] = 0
        context = hashlib.sha256(context_f.tobytes() + context_c.tobytes()).digest()
        return context, rows, frozen, present

    def _observe(self, state):
        context, rows, frozen, _ = state
        if self._context != context:
            self.reset()
            self._context = context
        self._boards.add(rows)
        self._freezes.add(frozen)

    def mask(self, obs):
        state = self._state(obs)
        self._observe(state)
        _, rows, frozen, present = state
        allowed = np.asarray(obs["action_mask"], dtype=bool).copy()
        self.last_blocked = {}
        move_start = self.layout.native["A_MOVE"]
        board_size = self.layout.native["N_BOARD"]
        for action in np.flatnonzero(allowed):
            action = int(action)
            if action != 2 and action < move_start:
                continue
            reason = None
            if self.reversible_actions >= self.max_reversible_actions:
                reason = "reversible_action_budget"
            elif action == 2:
                all_frozen = bool(present.any() and np.asarray(frozen)[present].all())
                after = tuple(bool(exists and not all_frozen) for exists in present)
                if after in self._freezes:
                    reason = "returns_to_seen_freeze_state"
            else:
                src, dst = divmod(action - move_start, board_size)
                moved = list(rows)
                moved.insert(dst, moved.pop(src))
                if tuple(moved) in self._boards:
                    reason = "returns_to_seen_board"
            if reason:
                allowed[action] = False
                self.last_blocked[action] = reason
        if not allowed.any():
            raise ValueError("No legal action remains; only call the filter for a ready player")
        return allowed

    def commit(self, action, before, after, *, done=False):
        """Record an executed action, never an unconfirmed recommendation."""
        if done:
            self.reset()
            return
        old, new = self._state(before), self._state(after)
        self._observe(old)
        if int(action) == 2 or int(action) >= self.layout.native["A_MOVE"]:
            if old[0] == new[0]:
                self.reversible_actions += 1
            else:
                self.reset()
        else:
            # A purchase/play/refresh/choice/end can justify a fresh arrangement
            # or Freeze decision, even when some visible fields happen to match.
            self.reset()
        self._observe(new)
