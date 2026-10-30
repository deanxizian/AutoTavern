"""Partial, provenance-preserving conversion from Power.log to policy features.

NaN / -1 mean unknown, never an empty slot. Only a complete observation may be
inferred. The coverage report makes remaining mechanic adapters explicit.
"""

from dataclasses import dataclass, field
import hashlib

import numpy as np

from .client_actions import bind_actions
from .client_common import CATALOG, load_catalog


RACES = ("BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECHANICAL", "MURLOC", "PIRATE", "QUILBOAR", "UNDEAD", "ABERRATION", "NAGA")
KEYWORDS = ("TAUNT", "DIVINE_SHIELD", "WINDFURY", "REBORN", "VENOMOUS", "POISONOUS", "STEALTH", "MAGNETIC", "MEGA_WINDFURY", "CLEAVE", "IMMUNE", "IMMUNE_ATTACKING")
KW_TAGS = ("TAUNT", "DIVINE_SHIELD", "WINDFURY", "REBORN", "VENOMOUS", "POISONOUS", "STEALTH", "MAGNETIC", "MEGA_WINDFURY", None, "IMMUNE", "IMMUNE_WHILE_ATTACKING")
GLOBAL_NAMES = ("turn", "hero_select", "game_over", "alive_count", "alive", "health", "armor", "gold",
    "max_gold_bonus", "gold_next_turn", "tier", "upgrade_price", "refresh_price", "minion_price", "free_refreshes",
    "frozen", "power0_usable", "power0_price", "power0_uses", "power1_usable", "power1_price", "power1_uses",
    "dark_gift_usable", "dark_gift_uses", "trinket_count", "deity_attack", "deity_health", "won_last", "lost_last",
    "spell_attack", "spell_health", "gem_attack", "gem_health", "board_count", "hand_count", "has_deity",
    "hand_queue_count", "next_is_ghost", "tavern_attack", "tavern_health", "effect_count", "chronum")
GLOBAL_SCALES = (20, 1, 1, 8, 1, 40, 20, 10, 5, 5, 6, 10, 2, 3, 3, 1, 1, 5, 5, 1, 5, 5, 1, 3, 2, 10, 10, 1, 1, 5, 5, 5, 5, 7, 10, 1, 5, 1, 10, 10, 5, 4)
FEATURE_ZONES = {"board": (68, 7, 31), "hand": (285, 10, 40), "shop": (685, 8, 40), "choices": (1005, 16, 40)}
CARD_ZONES = {"board": (19, 7), "hand": (26, 10), "shop": (36, 8), "choices": (44, 16)}


@dataclass
class ClientInput:
    frame_id: str
    features: np.ndarray
    cards: np.ndarray
    action_mask: np.ndarray
    actions: object
    gaps: list = field(default_factory=list)
    attachments: object = None
    growth: object = None
    strategy_view: dict | None = None
    scope: str = "client_recording"

    @property
    def complete(self):
        return not self.gaps and np.isfinite(self.features).all() and (self.cards >= 0).all() and self.action_mask.any()

    def report(self):
        return {"schema": "autotavern.model-input.v1", "frame_id": self.frame_id, "scope": self.scope,
                "status": "complete" if self.complete else "blocked", "phase": self.actions.phase,
                "known_features": int(np.isfinite(self.features).sum()), "total_features": 3227,
                "known_card_slots": int((self.cards >= 0).sum()), "total_card_slots": 124,
                "mapped_actions": np.flatnonzero(self.action_mask).tolist(), "gaps": self.gaps,
                "model_ready": False, "live_ready": False}


class ClientFeatureAdapter:
    """No game simulation and no guessed opponent boards or growth counters."""

    def __init__(self, requirements=None):
        self.meta, self.catalog = load_catalog()
        self.vocabulary = {cid: i + 1 for i, cid in enumerate(self.catalog)}
        self.catalog_sha256 = hashlib.sha256(CATALOG.read_bytes()).hexdigest()
        self.requirements = requirements if requirements is not None else {"attachments": True, "growth": True, "strategy": True}

    def convert(self, snapshot):
        if snapshot.get("public_history", {}).get("complete_from_start"):
            from .client_public import public_view, IncompletePublicState
            from .client_encoding import encode_public_view
            a = bind_actions(snapshot)
            try:
                view = public_view(snapshot, a, self.catalog)
                f, c, attachments, growth = encode_public_view(view, self.vocabulary)
                return ClientInput(a.frame_id, f, c, np.asarray(a.mask, dtype=np.bool_), a,
                    attachments=attachments, growth=growth, strategy_view=view)
            except IncompletePublicState as error:
                partial = self._partial(snapshot)
                partial.gaps = [{"code": str(error)}]
                return partial
        return self._partial(snapshot)

    def _partial(self, snapshot):
        a = bind_actions(snapshot)
        out = ClientInput(a.frame_id, np.full(3227, np.nan, np.float32), np.full(124, -1, np.int32),
                          np.asarray(a.mask, dtype=np.bool_), a, list(a.issues))
        f, c = out.features, out.cards
        def gap(code, **details):
            row = {"code": code, **details}
            if row not in out.gaps:
                out.gaps.append(row)
        if not a.player:
            return out
        source = snapshot.get("source", {})
        if snapshot.get("status") != "snapshot" or source.get("partial_trailing_line") or source.get("changed_while_reading"):
            gap("unstable_snapshot")
        if a.phase not in ("hero_select", "recruit"):
            gap("not_action_phase")
        entities = snapshot.get("entities", [])
        by_id = {e["entity_id"]: e for e in entities}
        me, hero = a.player.get("tags", {}), a.hero.get("tags", {})
        owner = a.player.get("controller")
        game = next((e.get("tags", {}) for e in entities if e.get("type") == "GAME"), {})
        own = [e for e in entities if e.get("controller") == owner and e.get("zone") == "PLAY"]
        buttons = {e.get("card_id"): e.get("tags", {}) for e in own if e.get("type") == "GAME_MODE_BUTTON"}
        def card_index(entity):
            cid = entity.get("card_id")
            if cid not in self.vocabulary:
                gap("card_identity_unmapped", card_id=cid, entity_id=entity.get("entity_id"))
                return -1
            return self.vocabulary[cid]
        def number(index, value, scale=1):
            if isinstance(value, (int, float, bool)) and np.isfinite(value):
                f[index] = value / scale
        def tags_for(cid):
            return buttons.get(cid, {})
        upgrade = next((v for k, v in buttons.items() if k and k.startswith("TB_BaconShopTechUp")), {})
        trinkets = [e for e in own if e.get("type") == "BATTLEGROUND_TRINKET" and not e.get("tags", {}).get("BACON_IS_POTENTIAL_TRINKET")]
        # These are direct fields. The remaining globals intentionally stay unknown.
        values = {"turn": me.get("TURN"), "hero_select": a.phase == "hero_select", "game_over": a.phase == "game_over",
            "alive": me.get("PLAYSTATE") == 1 if "PLAYSTATE" in me else None,
            "health": hero["HEALTH"] - hero.get("DAMAGE", 0) if "HEALTH" in hero else None,
            "armor": hero.get("ARMOR", 0) if a.hero else None,
            "gold": me["RESOURCES"] - me.get("RESOURCES_USED", 0) + me.get("TEMP_RESOURCES", 0) if "RESOURCES" in me else None,
            "tier": me.get("PLAYER_TECH_LEVEL"), "upgrade_price": upgrade.get("COST"),
            "refresh_price": tags_for("TB_BaconShop_8p_Reroll_Button").get("COST"),
            "frozen": bool(a.zones["shop"]) and all(e.get("tags", {}).get("FROZEN", 0) for e in a.zones["shop"]),
            "dark_gift_usable": 6 in a.bindings, "trinket_count": len(trinkets),
            "deity_attack": me.get("BACON_OLD_GOD_ATTACK"), "deity_health": me.get("BACON_OLD_GOD_HEALTH"),
            "board_count": len(a.zones["board"]), "hand_count": len(a.zones["hand"])}
        for slot in range(2):
            power = a.zones["powers"][slot] if slot < len(a.zones["powers"]) else None
            pt = power.get("tags", {}) if power else {}
            values[f"power{slot}_usable"] = 4 + slot in a.bindings if not a.choice_kind else None
            values[f"power{slot}_price"] = pt.get("COST", 0) if power else 0
            values[f"power{slot}_uses"] = pt.get("NUM_TIMES_HERO_POWER_USED_THIS_GAME") if power else 0
            c[1 + slot] = card_index(power) if power else 0
        for i, (name, scale) in enumerate(zip(GLOBAL_NAMES, GLOBAL_SCALES)):
            number(i, values.get(name), scale)
        # Complete ordered race/persistent-effect reconstruction is still required;
        # observing one tribe in the shop does not establish the lobby race pool.
        if a.phase == "hero_select":
            c[0] = 0
        else:
            c[0] = card_index(a.hero)
        deity_dbf = game.get("BACON_GLOBAL_OLD_GOD_DBID")
        deity = [cid for cid, row in self.catalog.items() if row.get("dbf") == deity_dbf]
        if len(deity) == 1:
            c[3] = self.vocabulary[deity[0]]
            f[35] = 1
        c[4] = card_index(a.choice_source) if a.choice_kind == "discover" else 0
        kind = {None: 0, "hero": 1, "discover": 2, "hero_power": 5, "trinket": 6}.get(a.choice_kind)
        if kind is not None:
            f[59:68] = 0
            f[59 + kind] = 1
        if len(trinkets) > 8:
            gap("trinket_slots_overflow")
        elif len(trinkets) > 1:
            gap("trinket_order_unresolved")
        else:
            c[60:68] = 0
            for i, e in enumerate(trinkets):
                c[60 + i] = card_index(e)

        def minion(e):
            tags = e.get("tags", {})
            data = self.catalog.get(e.get("card_id"), {})
            v = np.full(31, np.nan, np.float32)
            v[0], v[1] = 1, tags.get("ATK", 0) / 10
            if "HEALTH" in tags:
                v[2], v[3] = (tags["HEALTH"] - tags.get("DAMAGE", 0)) / 10, tags["HEALTH"] / 10
            if "TECH_LEVEL" in tags:
                v[4] = tags["TECH_LEVEL"] / 6
            if data:
                v[5] = bool(data.get("normal"))
            # Attachment scripts and shield charges are not the count of raw buffs.
            for i, tag in enumerate(KW_TAGS):
                if tag is not None:
                    v[8 + i] = bool(tags.get(tag, 0))
            # CARDRACE describes one race; dual/all-type and transformed races need
            # their own mapping. Catalog race names alone cannot prove current races.
            return v

        for zone, (start, limit, width) in FEATURE_ZONES.items():
            cs, _ = CARD_ZONES[zone]
            rows = a.zones[zone]
            if len(rows) > limit:
                gap("zone_overflow", zone=zone)
            for i in range(limit):
                begin = start + i * width
                if i >= len(rows):
                    f[begin:begin + width], c[cs + i] = 0, 0
                    continue
                e = rows[i]
                c[cs + i] = card_index(e)
                if zone == "board":
                    f[begin:begin + width] = minion(e)
                    continue
                if zone == "choices" and a.choice_kind in ("hero", "hero_power"):
                    f[begin:begin + width] = 0
                    f[begin] = 1
                    continue
                f[begin] = 1
                f[begin + 1] = e.get("type") == "MINION"
                if zone in ("hand", "choices"):
                    if zone != "choices" or a.choice_kind != "trinket":
                        f[begin + 2] = 0  # Native convention: no hand/discover price.
                else:
                    helpers = [x for x in own if (x.get("card_id") or "").startswith("TB_BaconShop_DragBuy") and x.get("tags", {}).get("2442") == e["entity_id"]]
                    # Require an Options target too; numeric link tag is only a cross-check.
                    option_targets = {child.get("entity_id") for option in (snapshot.get("last_options") or {}).get("options", [])
                                      if any(x["entity_id"] == option.get("entity_id") for x in helpers)
                                      for child in option.get("children", [])}
                    if len(helpers) == 1 and e["entity_id"] in option_targets and "COST" in helpers[0]["tags"]:
                        f[begin + 2] = helpers[0]["tags"]["COST"] / 5
                    f[begin + 3:begin + 6] = 0  # Native shop locked/temp/combined are always zero.
                    f[begin + 6] = bool(e.get("tags", {}).get("FROZEN", 0))
                if zone == "choices":
                    f[begin + 3:begin + 9] = 0
                if e.get("type") == "MINION":
                    f[begin + 9:begin + 40] = minion(e)
                else:
                    # Spell data is a dynamic tier/value, not printed card cost.
                    f[begin + 9:begin + 40] = 0
                    f[begin + 9] = 1
                    f[begin + 10] = np.nan
                    data = self.catalog.get(e.get("card_id"), {})
                    f[begin + 13] = data.get("tier", np.nan) / 6
        missing_globals = [name for i, name in enumerate(GLOBAL_NAMES) if not np.isfinite(f[i])]
        if missing_globals:
            gap("global_fields_unresolved", fields=missing_globals)
        for name, begin, end in (("lobby_races", 42, 53), ("quests", 53, 59),
                                 ("opponent_public_and_memory", 1645, 3227)):
            if not np.isfinite(f[begin:end]).all():
                gap("feature_group_unresolved", field=name)
        for zone, (start, limit, width) in FEATURE_ZONES.items():
            if not np.isfinite(f[start:start + limit * width]).all():
                gap("entity_features_unresolved", zone=zone)
        if (c < 0).any():
            gap("card_slots_unresolved", slots=np.flatnonzero(c < 0).tolist())
        # New trained actors consume these additional inputs; unknown is not []/0.
        for name, code in (("attachments", "public_attachments_unresolved"),
                           ("growth", "public_growth_unresolved"), ("strategy", "public_strategy_view_unresolved")):
            if self.requirements.get(name):
                gap(code)
        return out
