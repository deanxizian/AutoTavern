"""Build the trained actor's public view from client tags and event-time history.

Missing tags mean their protocol default only when CREATE_GAME was observed.
Lobby tribes are an explicit per-game input: Power.log does not contain the list.
"""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

from hearthstone.enums import Race
from .client_encoding import KEYWORDS, RACES


class IncompletePublicState(ValueError):
    pass


@lru_cache(maxsize=1)
def metadata():
    return json.loads((Path(__file__).parent / "data/client-tags-255790.json").read_text())["cards"]

GEM_DATA = {"BG20_GEM_Taunt": 1, "BG20_GEM_DivineShield": 2, "BG20_GEM_Reborn": 3}

@lru_cache(maxsize=1)
def dbf_index():
    from .client_common import load_catalog
    return {r["dbf"]: cid for cid, r in load_catalog()[1].items()}


def public_view(snapshot, actions, catalog):
    history = snapshot.get("public_history", {})
    if not history.get("complete_from_start") or snapshot.get("reset_seen"):
        raise IncompletePublicState("complete_game_history_required")
    races = snapshot.get("lobby_races")
    if not isinstance(races, list) or len(races) != 5 or len(set(races)) != 5 or not set(races) <= set(RACES):
        raise IncompletePublicState("lobby_races_required")
    # A server block can intentionally stay open while waiting for Discover.
    # The complete current menu, not BLOCK_END after the user's answer, is the
    # actionable boundary. Partial bytes and concurrent writes still block.
    status_ok = snapshot.get("status") == "snapshot" or (snapshot.get("status") == "incomplete_snapshot" and snapshot.get("open_blocks", 0) > 0)
    if not status_ok or snapshot.get("source", {}).get("partial_trailing_line") or snapshot.get("source", {}).get("changed_while_reading"):
        raise IncompletePublicState("unstable_snapshot")
    if actions.issues:
        raise IncompletePublicState(actions.issues[0]["code"])
    if actions.phase not in ("hero_select", "recruit"):
        raise IncompletePublicState("not_action_phase")
    defs = metadata()
    by_dbf = dbf_index()
    entities = {e["entity_id"]: e for e in snapshot["entities"]}
    tags, hero_tags = actions.player["tags"], actions.hero.get("tags", {})
    owner, pid = actions.player["controller"], snapshot["perspective"]["player_id"]
    turn = tags.get("TURN", 0)
    attached = defaultdict(list)
    for e in entities.values():
        if e.get("zone") == "PLAY" and e.get("type") == "ENCHANTMENT" and e.get("card_id"):
            attached[e["tags"].get("ATTACHED")].append(e)
    pe = attached[actions.player["entity_id"]]
    def ench(cid, key="TAG_SCRIPT_DATA_NUM_1"):
        return sum(e["tags"].get(key, 0) for e in pe if e["card_id"] == cid)
    def cid_of(e):
        cid = e.get("card_id")
        if cid in GEM_DATA: cid = "BG20_GEM"
        t = e.get("tags", {})
        parent = t.get("BACON_SKIN_PARENT_ID") or defs.get(cid, {}).get("tags", {}).get("BACON_SKIN_PARENT_ID")
        if parent:
            cid = by_dbf.get(parent)
        if cid not in catalog:
            raise IncompletePublicState("card_identity_unmapped:" + str(cid))
        return cid
    def card(e):
        cid = cid_of(e)
        return {**catalog[cid], "id": cid}
    def dbf_card(dbf):
        if not dbf:
            return None
        cid = by_dbf.get(dbf)
        if not cid:
            raise IncompletePublicState("card_dbf_unmapped:" + str(dbf))
        return {**catalog[cid], "id": cid}
    def script_attachments(e, enchants):
        result = []
        # Dark Gifts have client enchantment tokens but use their spell IDs in
        # the actor's attachment vocabulary. Repeats represent distinct stacks.
        for en in enchants:
            cid, et = en["card_id"], en["tags"]
            candidates = []
            if cid.startswith("BG36_MidGameEffect_"):
                candidates = [x for x in catalog if cid in (x+"e", x+"e2", x+"e3") and x.startswith("BG36_MidGameEffect_")]
                if not candidates and cid in catalog:
                    candidates = [cid]
                if not candidates:
                    raise IncompletePublicState("gift_attachment_unmapped:"+cid)
            elif et.get("MAGNETIC"):
                source = by_dbf.get(et.get("CREATOR_DBID")) or history.get("revealed", {}).get(str(et.get("CREATOR")))
                if not source or source not in catalog:
                    raise IncompletePublicState("magnetic_attachment_unmapped:"+cid)
                candidates = [source]
            if candidates:
                if len(candidates) != 1:
                    raise IncompletePublicState("attachment_ambiguous:"+cid)
                result.append(card({"card_id": candidates[0]}))
        return result
    def minion(e, index=attached):
        base, t = card(e), e["tags"]
        en = index[e["entity_id"]]
        scripts = script_attachments(e, en)
        d = defs.get(e["card_id"], {}).get("tags", {})
        keywords = []
        for kw in KEYWORDS:
            tag = {"IMMUNE_ATTACKING": "IMMUNE_WHILE_ATTACKING"}.get(kw, kw)
            if kw == "CLEAVE":
                active = kw in base["keywords"] or any(x["id"] == "BG36_MidGameEffect_000t25" for x in scripts)
            else:
                active = t.get(tag, d.get(tag, 0)) > 0
            if active: keywords.append(kw)
        current_races = list(base["races"])
        if t.get("CARDRACE"):
            try: primary = Race(t["CARDRACE"]).name
            except ValueError: raise IncompletePublicState("race_tag_unmapped")
            if primary == "ALL": current_races = list(RACES)
            elif primary not in current_races and primary in RACES: current_races = [primary]
        if "ALL" in current_races or any(x["id"] == "BG36_MidGameEffect_000t22" for x in scripts):
            current_races = list(RACES)
        for key, value in t.items():
            if key.startswith("RACE_ALSO_") and key != "RACE_ALSO_UPDATE_COUNT" and value:
                race = key.removeprefix("RACE_ALSO_")
                race = {"MECH": "MECHANICAL", "QUILLBOAR": "QUILBOAR"}.get(race, race)
                if race not in RACES: raise IncompletePublicState("dynamic_secondary_race_unmapped:"+race)
                if race not in current_races: current_races.append(race)
        shield_hits = 3 if any(x["id"] == "BG36_MidGameEffect_000t15" for x in scripts) else 1
        return {**base, "type": "minion", "eid": e["entity_id"], "atk": t.get("ATK", base["atk"]),
            "hp": t.get("HEALTH", base["hp"])-t.get("DAMAGE", 0), "max_hp": t.get("HEALTH", base["hp"]),
            "tier": t.get("TECH_LEVEL", base["tier"]), "golden": bool(base["normal"]),
            "keywords": keywords, "races": current_races, "attached": scripts,
            "shield_hits": shield_hits if "DIVINE_SHIELD" in keywords else 0,
            "activate_cost": t.get("INTERACTABLE_OBJECT_COST"), "counter": t.get("TAG_SCRIPT_DATA_NUM_1", 0),
            "shield_max": shield_hits, "attachment_data": [x.get("data", 0) for x in scripts],
            "doomed_turn": turn if any(x["card_id"] == "BG34_888e" for x in en) else 0}
    def item(e):
        if e.get("type") == "MINION":
            return minion(e)
        base, t = card(e), e["tags"]
        data = 0
        # Triple Reward's parameter is the offered tier, not its printed cost.
        if base["id"] == "TB_BaconShop_Triples_01":
            data = t.get("TAG_SCRIPT_DATA_NUM_1")
            if data is None: raise IncompletePublicState("triple_reward_tier_missing")
        if base["id"] == "BG20_GEM":
            # SDN1/2 are displayed Attack/Health, not the native variant enum.
            data = GEM_DATA.get(e["card_id"], 4 if t.get("BACON_TAVERN_SPELL") else 0)
        return {**base, "type": "spell", "data": data}
    power_counts = history.get("powers", {}).get(str(owner), {})
    from training.growth import FEATURE_NAMES
    stats = dict.fromkeys(FEATURE_NAMES, 0)
    for cid, names in (("BG25_008pe", ("eternal_knight_deaths", None, "knight_bonus")),
                       ("BG25_011pe", ("undead_atk", "undead_hp")),
                       ("BG31_808pe", ("beetle_atk", "beetle_hp")),
                       ("BG34_170pe", ("volumizer_atk", "volumizer_hp")),
                       ("BG_ShopBuff_Elemental", ("tavern_elemental_atk", "tavern_elemental_hp")),
                       ("BG31_816pe", ("baller_bonus",))):
        for i, name in enumerate(names):
            if name: stats[name] = ench(cid, "TAG_SCRIPT_DATA_NUM_"+str(i+1))
    stats.update(elem_give_atk=tags.get("BACON_ELEMENTAL_BUFFATKVALUE", 0),
        elem_give_hp=tags.get("BACON_ELEMENTAL_BUFFHEALTHVALUE", 0),
        lobster_bonus=tags.get("4803", 0), magnetized=tags.get("3670", 0),
        spell_atk=tags.get("TAVERN_SPELL_ATTACK_INCREASE", 0), spell_hp=tags.get("TAVERN_SPELL_HEALTH_INCREASE", 0),
        gem_atk=max(tags.get("BACON_BLOODGEMBUFFATKVALUE", 0), ench("BG26_159pe")),
        gem_hp=max(tags.get("BACON_BLOODGEMBUFFHEALTHVALUE", 0), ench("BG26_159pe", "TAG_SCRIPT_DATA_NUM_2")),
        tavern_atk=ench("BG_ShopBuff"), tavern_hp=ench("BG_ShopBuff", "TAG_SCRIPT_DATA_NUM_2"),
        won_last=bool(tags.get("BACON_WON_LAST_COMBAT", 0)),
        lost_last=tags.get("DAMAGE_DEALT_TO_HERO_LAST_TURN", 0) > 0,
        spells_cast=tags.get("NUM_SPELLS_PLAYED_THIS_GAME", 0), tavern_spells_cast=tags.get("3088", 0),
        battlecries_triggered=tags.get("3873", 0), deathrattles_triggered=tags.get("4639", 0),
        elementals_played=tags.get("2878", 0), golden_played=tags.get("4799", 0),
        gold_spent_total=tags.get("NUM_RESOURCES_SPENT_THIS_GAME", 0))
    stats.update(cards_played_turn=tags.get("NUM_CARDS_PLAYED_THIS_TURN", 0),
        minions_played_turn=tags.get("NUM_MINIONS_PLAYED_THIS_TURN", 0),
        gold_spent_turn=tags.get("RESOURCES_USED", 0),
        **history.get("stats", {}).get(str(owner), {}))
    stats["volumizer_atk"] = max(stats["volumizer_atk"], tags.get("4468", 0))
    stats["volumizer_hp"] = max(stats["volumizer_hp"], tags.get("4469", 0))
    stats["spellcraft_cast"] = sum(n for cid, n in power_counts.items() if "SPELLCRAFT" in catalog.get(cid, {}).get("keywords", [])
        or defs.get(cid, {}).get("tags", {}).get("SPELLCRAFT_HINT"))
    own = [e for e in entities.values() if e.get("controller") == owner and e.get("zone") == "PLAY"]
    buttons = {e.get("card_id"): e for e in own if e.get("type") == "GAME_MODE_BUTTON"}
    def button_cost(prefix, default):
        found = next((e for cid, e in buttons.items() if cid and cid.startswith(prefix)), None)
        if not found and prefix == "TB_BaconShopTechUp" and tags.get("PLAYER_TECH_LEVEL", 1) >= tags.get("BACON_MAX_PLAYER_TECH_LEVEL", 6):
            return int(any(e.get("card_id") == "TB_BaconShop_HP_054" for e in actions.zones["powers"]))
        if not found and actions.phase != "hero_select" and not (actions.choice_kind and not buttons):
            raise IncompletePublicState("button_missing:"+prefix)
        return found["tags"].get("COST", 0) if found else default
    trinkets = [e for e in own if e.get("type") == "BATTLEGROUND_TRINKET" and not e["tags"].get("BACON_IS_POTENTIAL_TRINKET")]
    trinkets.sort(key=lambda e: e["entity_id"])
    if len(trinkets) > 8: raise IncompletePublicState("trinket_overflow")
    powers = []
    from .client_native import scripts as native_scripts
    scripts = native_scripts()
    for i, e in enumerate(actions.zones["powers"]):
        t = e["tags"]
        power = card(e)
        sc = scripts.get(power["id"], {})
        uses = t.get("HEROPOWER_ACTIVATIONS_THIS_TURN", tags.get("HEROPOWER_ACTIVATIONS_THIS_TURN", 0) if len(actions.zones["powers"]) == 1 else 0)
        usable = sc.get("hero_power", False) and not t.get("EXHAUSTED", 0) and not t.get("HERO_POWER_DISABLED", 0) and uses < sc.get("hero_power_uses", 1)
        usable = usable and t.get("COST", 0) <= tags.get("RESOURCES", 0)-tags.get("RESOURCES_USED", 0)+tags.get("TEMP_RESOURCES", 0)
        powers.append({**power, "slot": i, "action": 4+i, "usable": 4+i in actions.bindings if not actions.choice_kind else usable,
            "price": t.get("COST", 0), "uses_game": tags.get("NUM_TIMES_HERO_POWER_USED_THIS_GAME", 0) if len(actions.zones["powers"]) == 1 else power_counts.get(e["card_id"], 0),
            "uses_turn": uses, "counter": t.get("TAG_SCRIPT_DATA_NUM_1", 0)})
    game = next(e["tags"] for e in entities.values() if e.get("type") == "GAME")
    deity_card = dbf_card(game.get("BACON_GLOBAL_OLD_GOD_DBID")) if "ABERRATION" in races else None
    me = {"hero": card(actions.hero) if actions.phase != "hero_select" else None,
        "health": hero_tags.get("HEALTH", 40)-hero_tags.get("DAMAGE", 0), "armor": hero_tags.get("ARMOR", 0),
        "alive": tags.get("PLAYSTATE", 1) == 1, "tier": tags.get("PLAYER_TECH_LEVEL", 1),
        "gold": tags.get("RESOURCES", 0)-tags.get("RESOURCES_USED", 0)+tags.get("TEMP_RESOURCES", 0),
        "max_gold_bonus": tags.get("3148", 10)-10,
        "gold_next_turn": ench("BG26_135e")-ench("BG28_513e"),
        "upgrade_price": button_cost("TB_BaconShopTechUp", 6),
        "refresh_price": button_cost("TB_BaconShop_8p_Reroll_Button", 1), "minion_price": 3,
        "free_refreshes": ench("Bacon_Free_Refresh_Player_Ench"),
        "frozen": any(e["tags"].get("FROZEN", 0) for e in actions.zones["shop"]),
        "hero_powers": powers, "board": [minion(e) for e in actions.zones["board"]],
        "hand": [], "shop": [], "trinkets": [card(e) for e in trinkets],
        "quests": [], "rewards": [], "secrets": [], "stats": stats, "pending": None,
        "hand_queue": [], "hand_limit": 10, "next_is_ghost": False, "effects": [],
        "chronum_saved": tags.get("BACON_BLACK_MARKET_EXTRA_CHRONUM", 0),
        "deity": {"card": deity_card, "atk": tags.get("BACON_OLD_GOD_ATTACK", 0), "hp": tags.get("BACON_OLD_GOD_HEALTH", 0)} if deity_card else None,
        "dark_gift": {"available": 6 in actions.bindings if not actions.choice_kind else bool(buttons.get("BG36_Button_DarkGift") and not buttons["BG36_Button_DarkGift"]["tags"].get("EXHAUSTED", 0) and tags.get("RESOURCES", 0)-tags.get("RESOURCES_USED", 0) >= buttons["BG36_Button_DarkGift"]["tags"].get("COST", 0)),
            "uses": power_counts.get("BG36_Button_DarkGift", 0)}}
    # Persistent listeners are keyed by source script, not by raw buff count.
    effect_counts = {}
    for cid, count in power_counts.items():
        if scripts.get(cid, {}).get("persistent") and not cid.startswith(("BG34_Giant_039", "BG34_Giant_201")):
            effect_counts[cid] = count * (2 if catalog.get(cid, {}).get("normal") else 1)
    for e in pe:
        source = by_dbf.get(e["tags"].get("CREATOR_DBID"))
        if not source:
            source = next((c for c, s in scripts.items() if (s["persistent"] or s["next_combat"] or s["one_pending"]) and e["card_id"] in (c+"e", c+"e2", c+"pe")), None)
        if source and (scripts.get(source, {}).get("persistent") or scripts.get(source, {}).get("next_combat") or scripts.get(source, {}).get("one_pending")):
            effect_counts[source] = 1 if source == "BG34_Giant_201" else max(effect_counts.get(source, 0), e["tags"].get("TAG_SCRIPT_DATA_NUM_1", 1))
    if any(m["id"].startswith("BG34_Giant_039") for m in me["board"]):
        effect_counts["BG34_Giant_039"] = 1
    me["effects"] = [{"card": {**catalog[c], "id": c}, "stacks": n} for c, n in effect_counts.items() if n > 0]
    for e in actions.zones["hand"]:
        t, en = e["tags"], {x["card_id"] for x in attached[e["entity_id"]]}
        locked = 0
        if t.get("LITERALLY_UNPLAYABLE"):
            if "BG22_HERO_201pe" in en:
                locked = 1  # Native held cards retain locked=1 until released.
            elif "BG27_HERO_801p2e" in en:
                locked = 1
            elif "TB_BaconShop_HP_068e3" in en:
                locked = next(x["tags"].get("TAG_SCRIPT_DATA_NUM_1", 1) for x in attached[e["entity_id"]] if x["card_id"] == "TB_BaconShop_HP_068e3")
            else:
                raise IncompletePublicState("hand_lock_unmapped")
        if t.get("DORMANT") or t.get("UNTOUCHABLE"):
            locked = max(locked, t.get("TAG_SCRIPT_DATA_NUM_1", 1))
        me["hand"].append({**item(e), "hid": e["entity_id"], "locked": locked, "temporary": "BG23_Spellcraft_e" in en,
            "combined": bool(t.get("CHOOSE_BOTH")), "doomed": "BG34_888e" in en,
            "link": t.get("BACON_LINKED_CARD_IN_HAND_ID", t.get("LINKED_ENTITY", 0)), "group": 0,
            "held": bool(en & {"BG22_HERO_201pe", "BG27_HERO_801p2e"}),
            "held_gold": next((x["tags"].get("TAG_SCRIPT_DATA_NUM_1", 60) for x in attached[e["entity_id"]] if x["card_id"] == "BG27_HERO_801p2e"), 0),
            "src": t.get("CREATOR", 0)})
    # Linked groups are connected components, distinct from a single paired card.
    hand_by_id = {h["hid"]: h for h in me["hand"]}
    neighbours = defaultdict(set)
    for h in me["hand"]:
        if h["link"] in hand_by_id:
            neighbours[h["hid"]].add(h["link"])
            neighbours[h["link"]].add(h["hid"])
    visited = set()
    for hid in hand_by_id:
        group, todo = set(), [hid]
        while todo:
            eid = todo.pop()
            if eid in group: continue
            group.add(eid); todo.extend(neighbours[eid]-group)
        if group & visited: continue
        visited |= group
        if len(group) > 2:
            for eid in group:
                hand_by_id[eid].update(group=min(group), group_left=max(1, len(group)-2), link=0)
    for e in actions.zones["shop"]:
        helpers = [h for h in own if (h.get("card_id") or "").startswith("TB_BaconShop_DragBuy") and h["tags"].get("2442") == e["entity_id"]]
        if len(helpers) != 1:
            raise IncompletePublicState("shop_price_unresolved")
        price = helpers[0]["tags"].get("COST", 0)
        me["shop"].append({**item(e), "sid": e["entity_id"], "price": price,
            "costs_health": bool(helpers[0]["tags"].get("BACON_COSTS_HEALTH_TO_BUY") or helpers[0]["tags"].get("CARD_COSTS_HEALTH") or e["tags"].get("BACON_COSTS_HEALTH_TO_BUY")),
            "frozen": bool(e["tags"].get("FROZEN")), "kept": e["entity_id"] in history.get("shop_at_combat", {}).get(str(e["controller"]), []),
            "ticket": any(x["card_id"] == "TB_BaconShop_HP_101e" for x in attached[e["entity_id"]])})
    if any(p["id"] in ("TB_BaconShop_HP_054", "TB_BaconShop_HP_014") for p in powers):
        me["minion_price"] = 2
    for e in own:
        t = e["tags"]
        if t.get("QUEST") and t.get("QUEST_REWARD_DATABASE_ID"):
            me["quests"].append({"quest": card(e), "reward": dbf_card(t["QUEST_REWARD_DATABASE_ID"]),
                "progress": t.get("QUEST_PROGRESS", 0), "goal": t.get("QUEST_PROGRESS_TOTAL", 1)})
    for e in entities.values():
        if e.get("controller") == owner and e.get("zone") == "SECRET" and e.get("card_id") in catalog:
            if e["tags"].get("SECRET"): me["secrets"].append(card(e))
            elif e["tags"].get("QUEST_REWARD"): me["rewards"].append(card(e))
    if len(me["quests"]) > 3 or len(me["secrets"]) > 5 or len(me["rewards"]) > 3:
        raise IncompletePublicState("effect_slots_overflow")
    if any(r["id"] == "BG27_Reward_811" for r in me["rewards"]) or any(x["card"]["id"] == "BGS_Treasures_022" for x in me["effects"]):
        me["minion_price"] = 2
    if any((p["id"] == "TB_BaconShop_HP_065t2" or (p["id"] == "TB_BaconShop_HP_065" and p["counter"] >= 14)) and p["uses_turn"] == 0 for p in powers):
        me["minion_price"] = 0
    if actions.choice_kind:
        me["pending"] = {"kind": actions.choice_kind, "options": []}
        for e in actions.zones["choices"]:
            o = item(e) if actions.choice_kind == "discover" else card(e)
            if actions.choice_kind == "trinket": o["cost"] = e["tags"].get("COST", o["cost"])
            if actions.choice_kind == "quest":
                o.update(reward=dbf_card(e["tags"]["QUEST_REWARD_DATABASE_ID"]), goal=e["tags"].get("QUEST_PROGRESS_TOTAL", 1))
            me["pending"]["options"].append(o)
    opponents = []
    hero_rows = {e["tags"].get("PLAYER_ID"): e for e in entities.values() if e.get("type") == "HERO"
        and e.get("zone") == "SETASIDE" and e.get("card_id") and e["tags"].get("PLAYER_ID") in range(1, 9)}
    for other in range(1, 9):
        if other == pid: continue
        e = hero_rows.get(other)
        if not e and actions.phase != "hero_select": raise IncompletePublicState("opponent_summary_missing")
        t = e["tags"] if e else {}
        seen = history.get("seen_boards", {}).get(str(other), {})
        mem = {x["entity_id"]: x for x in seen.get("entities", [])}
        en = defaultdict(list)
        for x in mem.values():
            if x["tags"].get("ATTACHED"): en[x["tags"]["ATTACHED"]].append(x)
        hp = t.get("HEALTH", 40)-t.get("DAMAGE", 0)
        dead = bool(t.get("BACON_DIED_LAST_COMBAT")) or hp <= 0
        opponents.append({"id": other-1, "hero": card(e) if e else None, "health": hp, "armor": t.get("ARMOR", 0),
            "tier": t.get("PLAYER_TECH_LEVEL", 1), "alive": not dead,
            "placement": t.get("PLAYER_LEADERBOARD_PLACE", 0) if dead else 0,
            "was_last_opponent": history.get("last_opponents", {}).get(str(pid)) == other,
            "is_next_opponent": tags.get("NEXT_OPPONENT_PLAYER_ID") == other,
            "seen_board": [minion(mem[i], en) for i in seen.get("board_ids", [])]})
    me["next_is_ghost"] = any(o["is_next_opponent"] and not o["alive"] for o in opponents)
    source = card(actions.choice_source) if actions.choice_kind == "discover" else None
    return {"schema_version": 1, "seat": pid-1, "phase": actions.phase, "turn": turn,
        "lobby_races": list(races), "alive": int(me["alive"])+sum(o["alive"] for o in opponents),
        "me": me, "opponents": opponents, "pending_source": source, "deity": deity_card,
        "paradox": dbf_card(game.get("BACON_GLOBAL_ANOMALY_DBID")), "mask": actions.mask}
