"""History-aware decoding for cancelled targeting, without forcing a target choice."""

from __future__ import annotations

import copy
import hashlib

import numpy as np


class CancelRetryGuard:
    """Do not reopen an exactly restored state/action after cancelling its targeting.

    Hashes include the unmodified player observation AND native legal mask. Only an
    action actually cancelled on the path back to the same visible state is excluded.
    Refreshing, freezing and rearranging are otherwise unchanged. Cancellation remains
    available, and no illegal fallback or fabricated target is introduced.
    """
    version = 1

    def __init__(self):
        self.states = {}
        self.prepared = {}

    def reset(self, identity):
        self.states.pop(int(identity), None)
        self.prepared.pop(int(identity), None)

    def apply(self, obs, identities=None):
        identities = np.arange(len(obs["ready"])) if identities is None else identities
        if len(identities) != len(obs["ready"]):
            raise ValueError("One identity is required per observation")
        masks = obs["action_mask"].copy()
        for row in np.flatnonzero(obs["ready"]):
            identity = int(identities[row])
            turn = round(float(obs["features"][row, 0]) * 20)
            state = self.states.get(identity)
            if state is None or state["turn"] != turn:
                state = self.states[identity] = {
                    "turn": turn, "history": [], "banned": {}, "cancelled": False}
            digest = hashlib.sha256(b"".join(obs[k][row].tobytes()
                                           for k in ("features", "cards", "action_mask"))).hexdigest()
            if state["cancelled"]:
                for prior, action in reversed(state["history"]):
                    if prior == digest and action != 7:
                        banned = state["banned"].setdefault(digest, [])
                        if action not in banned:
                            banned.append(action)
                        break
                state["cancelled"] = False
            for action in state["banned"].get(digest, []):
                masks[row, action] = False
            if not masks[row].any():
                raise RuntimeError("All actions led to cancelled targeting; inspect this state")
            self.prepared[identity] = (digest, masks[row].copy())
        return {**obs, "action_mask": masks}

    def record(self, actions, identities=None):
        identities = range(len(actions)) if identities is None else identities
        if len(identities) != len(actions):
            raise ValueError("One identity is required per recorded action")
        for action, identity in zip(actions, identities):
            identity, action = int(identity), int(action)
            digest, mask = self.prepared[identity]
            if action < 0 or action >= len(mask) or not mask[action]:
                raise ValueError("Action is not allowed by the prepared decoder mask")
            state = self.states[identity]
            state["history"].append([digest, action])
            state["history"] = state["history"][-32:]
            state["cancelled"] = action == 7

    def save(self, identity):
        return {"version": self.version, "state": copy.deepcopy(self.states.get(int(identity)))}

    def load(self, identity, data):
        if data["version"] != self.version:
            raise ValueError("Unsupported cancellation decoder version")
        self.reset(identity)
        if data["state"] is not None:
            self.states[int(identity)] = copy.deepcopy(data["state"])
