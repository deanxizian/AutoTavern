"""Bind the 245 policy actions to a particular public client snapshot.

This module never sends game input. An Options target is not a model Choose
index: targeted plays remain explicitly unresolved until their second decision
has an exact observation. Shop dragging has no counterpart in the policy.
"""

from dataclasses import dataclass, field

from .client_common import content_id


N_ACTIONS = 245


@dataclass
class ActionMap:
    frame_id: str
    phase: str = "unknown"
    player: dict = field(default_factory=dict)
    hero: dict = field(default_factory=dict)
    zones: dict = field(default_factory=lambda: {k: [] for k in ("board", "hand", "shop", "powers", "choices")})
    bindings: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    choice_kind: str | None = None
    choice_source: dict = field(default_factory=dict)
    unresolved_options: dict = field(default_factory=dict)

    @property
    def mask(self):
        return [i in self.bindings for i in range(N_ACTIONS)]

    def issue(self, code, **details):
        item = {"code": code, **details}
        if item not in self.issues:
            self.issues.append(item)


def frame_identity(snapshot):
    # Include packet freshness and completion, which can change without new tags.
    source = snapshot.get("source", {})
    return content_id({"snapshot": snapshot.get("snapshot_id"),
        "session": [source.get(k) for k in ("path", "device", "inode", "game_start_line")],
        "status": snapshot.get("status"), "game_status": snapshot.get("game_status"),
        "lobby_races": snapshot.get("lobby_races"),
        "options": snapshot.get("last_options"), "choices": snapshot.get("last_choices"),
        "options_current": snapshot.get("options_after_last_state_change"),
        "choices_current": snapshot.get("choices_after_last_state_change"),
        "choice_packet_complete": snapshot.get("choice_packet_complete"),
        "options_packet_complete": snapshot.get("options_packet_complete")})


def bind_actions(snapshot):
    out = ActionMap(frame_identity(snapshot))
    entities = {e["entity_id"]: e for e in snapshot.get("entities", [])}
    pid = snapshot.get("perspective", {}).get("player_id")
    players = [e for e in entities.values() if e.get("type") == "PLAYER"]
    own = [e for e in players if pid is not None and e.get("tags", {}).get("PLAYER_ID") == pid]
    if len(own) != 1:
        out.issue("local_player_unresolved")
        return out
    out.player = own[0]
    owner = out.player.get("controller")
    out.hero = entities.get(out.player.get("tags", {}).get("HERO_ENTITY"), {})
    game = next((e for e in entities.values() if e.get("type") == "GAME"), {})
    combat = game.get("tags", {}).get("BACON_IN_COMBAT_PHASE") == 1
    out.phase = "game_over" if snapshot.get("game_status") == "complete" else "combat" if combat else "recruit"
    if out.phase in ("game_over", "combat"):
        return out
    shops = {p.get("controller") for p in players if p != out.player and
             entities.get(p.get("tags", {}).get("HERO_ENTITY"), {}).get("card_id") == "TB_BaconShopBob"}
    for e in entities.values():
        zone, kind = e.get("zone"), e.get("type")
        if e.get("controller") == owner:
            dest = "hand" if zone == "HAND" and kind != "HERO" else "board" if zone == "PLAY" and kind == "MINION" else "powers" if zone == "PLAY" and kind == "HERO_POWER" else None
        else:
            dest = "shop" if zone == "PLAY" and e.get("controller") in shops and kind in ("MINION", "BATTLEGROUND_SPELL") else None
        if dest:
            out.zones[dest].append(e)
    for zone, limit in (("board", 7), ("hand", 10), ("shop", 8)):
        rows = out.zones[zone]
        rows.sort(key=lambda e: (e.get("position") or 0, e["entity_id"]))
        if len(rows) > limit or [e.get("position") for e in rows] != list(range(1, len(rows) + 1)):
            out.issue("zone_order_unresolved", zone=zone)
    # HERO_POWER DBF identifies the primary power. Never order two powers by EID.
    powers = out.zones["powers"]
    if len(powers) > 1:
        pt, ht = out.player.get("tags", {}), out.hero.get("tags", {})
        primary = pt.get("HERO_POWER_ENTITY") or ht.get("HERO_POWER_ENTITY")
        secondary = ht.get("ADDITIONAL_HERO_POWER_ENTITY_1") or pt.get("ADDITIONAL_HERO_POWER_ENTITY_1")
        if primary in {e["entity_id"] for e in powers}:
            powers.sort(key=lambda e: e["entity_id"] != primary)
        elif secondary in {e["entity_id"] for e in powers}:
            powers.sort(key=lambda e: e["entity_id"] == secondary)
        elif sum(bool(e.get("tags", {}).get("ADDITIONAL_HERO_POWER_INDEX")) for e in powers) == 1:
            powers.sort(key=lambda e: e.get("tags", {}).get("ADDITIONAL_HERO_POWER_INDEX", 0))
        else:
            out.issue("power_slot_order_unresolved")
    if len(powers) > 2:
        out.issue("power_slots_overflow")

    choice = snapshot.get("last_choices") or {}
    if choice and not choice.get("selection_sent_or_resolved", True):
        if choice.get("player_entity_id") != out.player["entity_id"]:
            out.issue("choice_player_mismatch")
            return out
        ids = choice.get("entity_ids", [])
        if choice.get("min") != 1 or choice.get("max") != 1 or not 0 < len(ids) <= 16 or len(set(ids)) != len(ids):
            out.issue("unsupported_choice_cardinality")
            return out
        out.zones["choices"] = [entities.get(eid, {}) for eid in ids]
        types = {e.get("type") for e in out.zones["choices"]}
        out.choice_source = entities.get(choice.get("source_entity_id"), {})
        out.choice_kind = ("hero" if types == {"HERO"} and choice.get("type") == "MULLIGAN" else
            "hero_power" if types == {"HERO_POWER"} else "trinket" if types == {"BATTLEGROUND_TRINKET"} else
            "quest" if all(e.get("tags", {}).get("QUEST") and e["tags"].get("QUEST_REWARD_DATABASE_ID") for e in out.zones["choices"]) else
            "discover" if types <= {"MINION", "BATTLEGROUND_SPELL", "SPELL"} else None)
        if out.choice_kind == "hero":
            out.phase = "hero_select"
        if not out.choice_kind:
            out.issue("choice_kind_unresolved")
        if snapshot.get("choice_packet_complete") is False or (not snapshot.get("choices_after_last_state_change") and not snapshot.get("choice_packet_complete")):
            out.issue("stale_choices")
        for i, e in enumerate(out.zones["choices"]):
            if not e.get("card_id"):
                out.issue("choice_identity_unresolved", slot=i)
                continue
            if out.choice_kind == "hero" and e.get("tags", {}).get("BACON_LOCKED_MULLIGAN_HERO"):
                continue  # Preserve displayed slots, but never recommend a locked offer.
            out.bindings[30 + i] = {"kind": "choose_" + (out.choice_kind or "unknown"),
                "source_entity_id": e["entity_id"], "card_id": e["card_id"], "choice_id": choice["id"],
                "position": i + 1, "name": e.get("catalog_name_zh") or e.get("catalog_name"), "zone": "option"}
        return out

    if not shops:
        out.issue("recruit_shop_unresolved")
    if not snapshot.get("options_after_last_state_change") or snapshot.get("options_packet_complete") is False:
        out.issue("stale_options")
    packet = snapshot.get("last_options") or {}
    if not packet.get("options"):
        out.issue("missing_options")
        return out
    index = {e["entity_id"]: (zone, i) for zone, rows in out.zones.items() for i, e in enumerate(rows)}
    entity_options = {}
    for opt in packet["options"]:
        entity_options.setdefault(opt.get("entity_id"), []).append(opt)
    # End means wait for combat; END_TURN's INVALID client button is not clicked.
    out.bindings[0] = {"kind": "wait_for_combat", "zone": "control"}
    for opt in packet["options"]:
        if opt.get("error") is not None or opt.get("type") == "END_TURN":
            continue
        e = entities.get(opt.get("entity_id"), {})
        cid, tags = e.get("card_id") or "", e.get("tags", {})
        zone, slot = index.get(e.get("entity_id"), (None, None))
        children = opt.get("children", [])
        targets = [c["entity_id"] for c in children if c.get("kind") == "target" and c.get("error") is None]
        binding = {"options_id": packet.get("id"), "option_id": opt["id"],
                   "source_entity_id": e.get("entity_id"), "card_id": cid,
                   "name": e.get("catalog_name_zh") or e.get("catalog_name"), "zone": zone or "control"}
        modes = [c for c in children if c.get("kind") == "subOption" and c.get("error") is None]
        if any(c.get("kind") not in ("target", "subOption") for c in children):
            out.issue("suboptions_unresolved", entity_id=e.get("entity_id")); continue
        if modes:
            binding["modes"] = [{"id": c["id"], "entity_id": c["entity_id"],
                "card_id": entities.get(c["entity_id"], {}).get("card_id"),
                "name": entities.get(c["entity_id"], {}).get("catalog_name_zh"), "children": c.get("children", [])} for c in modes]
        def add(action, kind, **values):
            if action in out.bindings:
                existing = out.bindings[action]
                candidate = {**binding, "kind": kind, **values}
                if {k: v for k, v in candidate.items() if k not in ("option_id", "option_ids")} == {k: v for k, v in existing.items() if k not in ("option_id", "option_ids")}:
                    existing.setdefault("option_ids", [existing["option_id"]]).append(opt["id"])
                    return  # Same human operation, two protocol aliases.
                out.issue("ambiguous_action", action=action)
                out.unresolved_options[opt["id"]] = "ambiguous_action"
                out.unresolved_options[out.bindings[action]["option_id"]] = "ambiguous_action"
                return  # Never silently replace an earlier client mode.
            out.bindings[action] = {**binding, "kind": kind, **values}
        if cid.startswith("TB_BaconShop_DragBuy") or cid == "TB_BaconShop_DragSell":
            buying = cid != "TB_BaconShop_DragSell"
            for target in targets:
                tz, ti = index.get(target, (None, None))
                if tz != ("shop" if buying else "board"):
                    out.issue("transaction_target_unresolved", entity_id=target)
                    continue
                t = entities[target]
                add((8 if buying else 16) + ti, "buy" if buying else "sell", target_entity_id=target,
                    target_card_id=t.get("card_id"), name=t.get("catalog_name_zh") or t.get("catalog_name"),
                    zone=tz, position=ti + 1, cost=tags.get("COST") if buying else None,
                    costs_health=bool(tags.get("BACON_COSTS_HEALTH_TO_BUY") or tags.get("CARD_COSTS_HEALTH") or t.get("tags", {}).get("BACON_COSTS_HEALTH_TO_BUY")))
        elif cid == "TB_BaconShop_8p_Reroll_Button":
            add(1, "refresh", cost=tags.get("COST"))
        elif cid == "TB_BaconShopLockAll_Button":
            frozen = any(e.get("tags", {}).get("FROZEN", 0) for e in out.zones["shop"])
            add(2, "unfreeze" if frozen else "freeze")
        elif cid.startswith("TB_BaconShopTechUp"):
            add(3, "upgrade", cost=tags.get("COST"))
        elif cid == "BG36_Button_DarkGift":
            add(6, "dark_gift", cost=tags.get("COST"))
        elif zone == "powers" and slot < 2:
            add(4 + slot, "hero_power", position=slot + 1, targets=targets)
        elif zone == "hand":
            if e.get("type") == "MINION":
                if tags.get("MAGNETIC"):
                    for target in out.zones["board"]:
                        # The client target list is authoritative; no mechanical
                        # tribe guess can manufacture an available magnetic target.
                        if target["entity_id"] in targets:
                            add(126+7*slot+target["position"]-1, "magnetize", hand_position=slot+1,
                                position=target["position"], target_entity_id=target["entity_id"])
                for pos in range(min(len(out.zones["board"]) + 1, 8)) if len(out.zones["board"]) < 7 else ():
                    add(46 + 8 * slot + pos, "play_minion", hand_position=slot + 1, position=pos + 1, targets=targets)
            else:
                add(46 + 8 * slot, "cast", position=slot + 1, targets=targets)
        elif zone == "board":
            siblings = entity_options[e["entity_id"]]
            # Options are not in a stable order: the same Bellringer's target
            # menu appears before OR after reposition in this recording.
            # Target children identify Activate. For an untargeted Activate,
            # the two empty legal entries are aliases until selectedPosition
            # distinguishes activation (0) from reposition (1..7).
            empty = [o for o in siblings if o.get("error") is None and not o.get("children")]
            targeted = any(o.get("error") is None and o.get("children") for o in siblings)
            moving = not children
            activating = tags.get("INTERACTABLE_OBJECT") and (bool(children) or (not targeted and len(empty) >= 2))
            if activating:
                cost = tags.get("INTERACTABLE_OBJECT_COST", 0)
                gold = out.player["tags"].get("RESOURCES", 0)-out.player["tags"].get("RESOURCES_USED", 0)+out.player["tags"].get("TEMP_RESOURCES", 0)
                if cost <= gold:
                    add(23+slot, "activate", position=slot+1, cost=cost, targets=targets)
            for pos in range(len(out.zones["board"])) if moving else ():
                if pos != slot and game.get("tags", {}).get("ALLOW_MOVE_MINION"):
                    add(196 + 7 * slot + pos, "move", from_position=slot + 1, position=pos + 1)
        elif zone == "shop":
            continue  # Client-only shop reposition, never a policy Move action.
        else:
            out.issue("unmapped_legal_option", entity_id=e.get("entity_id"), card_id=cid)
    return out


def render_bound_action(mapping, action):
    """Render only a bound, resolved operation. No arbitrary target fallback."""
    if type(action) is not int or action not in mapping.bindings:
        raise ValueError("Action is not bound to this frame")
    b = mapping.bindings[action]
    if b.get("option_id") in mapping.unresolved_options:
        raise ValueError("Client action mode is unresolved")
    if b.get("targets") and not b.get("resolved_target"):
        raise ValueError("Targeted action needs its exact second-stage model observation")
    if b.get("modes") and not b.get("resolved_mode"):
        raise ValueError("Choose One needs its exact second-stage model observation")
    kind, name = b["kind"], b.get("name")
    label = {"board": "我方战场", "hand": "手牌", "shop": "酒馆", "option": "选项", "powers": "英雄技能"}.get(b["zone"], "")
    ref = f"{label}第 {b.get('hand_position', b.get('position'))} 位「{name}」"
    if kind in ("buy", "sell", "cast", "hero_power") or kind.startswith("choose_") or kind in ("play_minion", "move"):
        if not name:
            raise ValueError("Missing display identity")
    if kind.startswith("choose_"):
        instruction = f"选择{ref}。"
    else:
        instruction = {"buy": f"购买{ref}。", "sell": f"出售{ref}。", "cast": f"打出{ref}。",
            "hero_power": f"使用{ref}。", "play_minion": f"打出{ref}，放在战场第 {b.get('position')} 位。",
            "activate": f"发动{ref}。", "magnetize": f"将手牌第 {b.get('hand_position')} 位「{name}」磁力合体到战场第 {b.get('position')} 位。",
            "move": f"将战场第 {b.get('from_position')} 位「{name}」移动到第 {b.get('position')} 位。",
            "refresh": "刷新一次酒馆。", "freeze": "冻结酒馆中的卡牌。", "unfreeze": "解冻酒馆中的卡牌。",
            "upgrade": "升级一次酒馆。", "dark_gift": "使用一次黑暗之赐。",
            "wait_for_combat": "本回合不再操作，等待战斗开始。"}.get(kind)
    if not instruction:
        raise ValueError("Unsupported bound operation")
    if b.get("resolved_mode"):
        instruction += f"选择「{b['resolved_mode']['name']}」。"
    if b.get("resolved_target"):
        t = b["resolved_target"]
        instruction += f"目标：{t['label']}第 {t['position']} 位「{t['name']}」。"
    return {"frame_id": mapping.frame_id, "action": action, "kind": kind, "instruction": instruction,
            "binding": dict(b), "execution_verified": False}


def match_submission(mapping, event):
    """Match actual client input against its preceding frame, not a proposal.

    This proves identity/order only. Server state changes must separately confirm
    execution before decoder history can advance.
    """
    result = {"frame_id": mapping.frame_id, "sequence": event.get("sequence"),
              "kind": event.get("kind"), "execution_verified": False}
    if event.get("kind") == "shop_position":
        return {**result, "status": "client_only_shop_move"}
    if event.get("kind") == "board_position" and event.get("position") == event.get("main", {}).get("position"):
        return {**result, "status": "client_only_noop"}
    if event.get("kind") != "choice" and event.get("option") in mapping.unresolved_options:
        return {**result, "status": "requires_mode_mapping", "reason": mapping.unresolved_options[event["option"]]}
    matches = []
    for action, binding in mapping.bindings.items():
        if event.get("kind") == "choice":
            if binding.get("choice_id") == event.get("choices_id") and event.get("entity_ids") == [binding.get("source_entity_id")]:
                matches.append(action)
            continue
        if (binding.get("options_id") != event.get("options_id") or event.get("option") not in binding.get("option_ids", [binding.get("option_id")])
                or binding.get("source_entity_id") != event.get("main", {}).get("entity_id")):
            continue
        target = event.get("target", {}).get("entity_id")
        targets = binding.get("targets", [])
        if binding.get("modes"):
            mode = next((m for m in binding["modes"] if m["id"] == event.get("suboption")), None)
            if mode is None:
                continue
            targets = [c["entity_id"] for c in mode["children"] if c["kind"] == "target" and c.get("error") is None]
        if binding.get("target_entity_id") is not None and binding["target_entity_id"] != target:
            continue
        if targets and target not in targets:
            continue
        if not targets and not binding.get("target_entity_id") and target not in (None, 0):
            continue
        if binding["kind"] in ("play_minion", "move") and binding["position"] != event.get("position"):
            continue
        if binding["kind"] == "activate" and event.get("position", 0) != 0:
            continue
        matches.append(action)
    if len(matches) != 1:
        return {**result, "status": "unmapped" if not matches else "ambiguous", "matches": matches}
    action = matches[0]
    binding = mapping.bindings[action]
    status = "requires_mode_decision" if binding.get("modes") else "requires_target_decision" if binding.get("targets") else "bound"
    return {**result, "status": status,
            "action": action, "target_entity_id": event.get("target", {}).get("entity_id")}
