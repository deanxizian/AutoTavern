"""Resolve client target menus through the actor's second decision, by entity ID."""
from copy import deepcopy
import numpy as np
from .client_encoding import encode_public_view


def pending_observation(packet, binding, vocabulary):
    view = deepcopy(packet.strategy_view)
    me = view["me"]
    kind = binding["kind"]
    # Spells/powers/Activate defer payment; battlecries first commit the summon.
    if kind not in ("cast", "hero_power", "activate", "play_minion"):
        raise ValueError("Unsupported target continuation")
    source = next((c for z in ("hand", "board", "hero_powers") for c in me[z]
                   if c.get("hid", c.get("eid")) == binding["source_entity_id"]), None)
    if kind == "hero_power": source = me["hero_powers"][binding["position"]-1]
    if source is None: raise ValueError("Missing source for target decision")
    cancellable = True
    if kind == "play_minion":
        from .client_native import request
        projection = request(op="play", view=view, binding=binding)
        projected = projection["me"]
        # Only the own-player consequence is hypothetical. Preserve actual public
        # opponents, lobby and client legality; never use a generated opponent.
        for key in ("board", "hand", "shop", "hand_queue", "stats", "effects", "health", "armor", "gold"):
            me[key] = projected[key]
        if projection.get("pending_source"):
            source = projection["pending_source"]
        cancellable = False  # Native Battlecry is already committed.
        view["continuation_projection"] = {"source": "native_public_transition", "observed": False,
            "random_transition": projection["random_transition"]}
    view["pending_source"] = source
    mask = np.zeros(245, np.bool_)
    mask[7] = cancellable
    modes = binding.get("modes")
    if modes:
        if len(modes) != 2 or [m["id"] for m in modes] != [0, 1]:
            raise ValueError("Unmapped Choose One ordering")
        me["pending"] = {"kind": "choose_one", "source": source, "options": [source, source]}
        mask[30:32] = True
        candidates = modes
    else:
        allowed = set(binding["targets"])
        candidates = []
        for zone, identity, label in (("board", "eid", "我方战场"), ("shop", "sid", "酒馆"), ("hand", "hid", "手牌")):
            for i, card in enumerate(me[zone]):
                if card.get(identity) in allowed:
                    candidates.append({"entity_id": card[identity], "position": i+1,
                        "label": label, "name": card.get("name_zh") or card.get("name"), "card": card})
        if {c["entity_id"] for c in candidates} != allowed or not 0 < len(candidates) <= 16:
            raise ValueError("Target is outside the public observation")
        me["pending"] = {"kind": "target", "targets": [c["entity_id"] for c in candidates]}
        mask[30:30+len(candidates)] = True
    view["mask"] = mask.tolist()
    f, c, attachments, growth = encode_public_view(view, vocabulary)
    obs = {"features": f[None], "cards": c[None], "action_mask": mask[None],
        "attachments": attachments, "growth": growth}
    return obs, view, candidates
