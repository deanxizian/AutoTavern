"""Policy v1 encoding of an explicit public view; no simulated/private state."""
import numpy as np

RACES = ("BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECHANICAL", "MURLOC", "PIRATE", "QUILBOAR", "UNDEAD", "ABERRATION", "NAGA")
KEYWORDS = ("TAUNT", "DIVINE_SHIELD", "WINDFURY", "REBORN", "VENOMOUS", "POISONOUS", "STEALTH", "MAGNETIC", "MEGA_WINDFURY", "CLEAVE", "IMMUNE", "IMMUNE_ATTACKING")


def encode_public_view(view, vocabulary):
    """Match obs.rs, including its non-obvious spell/hand/choice conventions."""
    from training.attachments import AttachmentRows
    from training.growth import from_views
    f, c = np.zeros(3227, np.float32), np.zeros(124, np.int32)
    bags = [[] for _ in range(124)]
    me, pending = view["me"], view["me"].get("pending") or {}
    stats = me["stats"]
    def ci(card):
        return vocabulary[card["id"]] if card else 0
    def card(slot, value):
        c[slot] = ci(value)
        bags[slot] = [ci(x) for x in ((value or {}).get("attached") or [])]
    def minion(start, m):
        if not m:
            return
        races = set(m["races"])
        f[start:start+31] = [1, m["atk"]/10, m["hp"]/10, m["max_hp"]/10,
            m["tier"]/6, m["golden"], len(m["attached"])/3, m["shield_hits"]/3,
            *[k in m["keywords"] for k in KEYWORDS], *[r in races or "ALL" in races for r in RACES]]
    def item(start, it, price=0, flags=(0, 0, 0, 0, 0, 0)):
        if it is None:
            return
        kind = it.get("type") or ("minion" if "max_hp" in it else None)
        if kind not in ("minion", "spell"):
            f[start] = 1
            return
        f[start:start+9] = [1, kind == "minion", price/5, *flags]
        if kind == "minion":
            minion(start+9, it)
        else:
            f[start+9:start+14] = [1, it["data"]/6, 0, 0, it["tier"]/6]
    powers = me["hero_powers"]
    ps = [(p["usable"], (p["price"] or 0)/5, p["uses_game"]/5) for p in powers]
    ps += [(0, 0, 0)] * (2-len(ps))
    deity = me.get("deity")
    f[:42] = [view["turn"]/20, view["phase"] == "hero_select", view["phase"] == "game_over",
        view["alive"]/8, me["alive"], me["health"]/40, me["armor"]/20, me["gold"]/10,
        me["max_gold_bonus"]/5, me["gold_next_turn"]/5, me["tier"]/6, me["upgrade_price"]/10,
        me["refresh_price"]/2, me["minion_price"]/3, me["free_refreshes"]/3, me["frozen"],
        *ps[0], *ps[1], me["dark_gift"]["available"], me["dark_gift"]["uses"]/3,
        len(me["trinkets"])/2, (deity["atk"] if deity else 0)/10, (deity["hp"] if deity else 0)/10,
        stats["won_last"], stats["lost_last"], stats["spell_atk"]/5, stats["spell_hp"]/5,
        stats["gem_atk"]/5, stats["gem_hp"]/5, len(me["board"])/7, len(me["hand"])/10,
        bool(deity), len(me["hand_queue"])/5, me["next_is_ghost"], stats["tavern_atk"]/10,
        stats["tavern_hp"]/10, len(me["effects"])/5, (pending["chronum"] if pending.get("kind") == "timewarp" else me["chronum_saved"])/4]
    f[42:53] = [r in view["lobby_races"] for r in RACES]
    for i, q in enumerate(me["quests"]):
        f[53+2*i:55+2*i] = [1, q["progress"]/max(q["goal"], 1)]
        card(5+3*i, q["quest"])
        card(6+3*i, q["reward"])
    for i, r in enumerate(me["rewards"]):
        card(7+3*i, r)
    kinds = (None, "hero", "discover", "choose_one", "target", "hero_power", "trinket", "quest", "timewarp")
    f[59+kinds.index(pending.get("kind"))] = 1
    card(0, me["hero"])
    for i, p in enumerate(powers): card(1+i, p)
    card(3, deity["card"] if deity else None)
    card(4, view.get("pending_source"))
    for i, s in enumerate(me["secrets"]): card(14+i, s)
    for i, m in enumerate(me["board"]):
        minion(68+31*i, m); card(19+i, m)
    for i, h in enumerate(me["hand"]):
        item(285+40*i, h, flags=(h["locked"], h["temporary"], h["combined"], h.get("doomed", False), 0, bool(h["group"])))
        card(26+i, h)
    for i, s in enumerate(me["shop"]):
        item(685+40*i, s, s["price"], (0, 0, 0, s["frozen"], s["kept"], s["ticket"]))
        card(36+i, s)
    opts = pending.get("options", [])
    if pending.get("kind") == "target":
        lookup = {x[field]: x for zone, field in (("board", "eid"), ("shop", "sid"), ("hand", "hid")) for x in me[zone]}
        opts = [lookup[eid] for eid in pending["targets"]]
    for i, o in enumerate(opts):
        if pending["kind"] in ("hero", "hero_power", "choose_one"):
            f[1005+40*i] = 1
        else:
            price = o["cost"] if pending["kind"] == "trinket" else o["goal"] if pending["kind"] == "quest" else o["chronum_cost"] if pending["kind"] == "timewarp" else 0
            if pending["kind"] == "quest":
                o = o["reward"]  # Native choice encodes reward, with quest goal as price.
            it = {**o, "type": "spell", "data": 0} if pending["kind"] in ("trinket", "quest") else o
            item(1005+40*i, it, price)
        card(44+i, o)
    if pending.get("kind") == "timewarp":
        f[1005+40*pending["leave"]] = 1
    for i, t in enumerate(me["trinkets"]): card(60+i, t)
    opponents = {o["id"]: o for o in view["opponents"]}
    for i in range(7):
        o = opponents[(view["seat"]+i+1) % 8]
        if view["phase"] == "hero_select":
            f[1645+9*i:1654+9*i] = [1, o["alive"], 1, 0, 1/6, 0, 0, 0, 0]
        else:
            f[1645+9*i:1654+9*i] = [1, o["alive"], o["health"]/40, o["armor"]/20,
                o["tier"]/6, len(o["seen_board"])/7, o["placement"]/8, o["was_last_opponent"], o["is_next_opponent"]]
            card(68+i, o["hero"])
        for j, m in enumerate(o["seen_board"]):
            minion(1708+(i*7+j)*31, m); card(75+i*7+j, m)
    return f, c, AttachmentRows(np.array([[len(b) for b in bags]], np.int32),
        np.array([x for b in bags for x in b], np.int32)), from_views([view])
