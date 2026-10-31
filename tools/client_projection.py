"""Display recorded ownership and zones without creating a policy observation."""


GROUPS = {
    "choices": "当前选择", "board": "我方战场", "hand": "我方手牌",
    "shop": "酒馆", "opponent": "当前对手 · 战斗场面",
    "player": "我方英雄与饰品", "other": "其他可见实体 · 归属待确认",
}
INTERNAL_TYPES = {"GAME", "PLAYER", "ENCHANTMENT", "GAME_MODE_BUTTON", "MOVE_MINION_HOVER_TARGET"}


def project_live_cards(snapshot):
    entities = snapshot.get("entities", [])
    by_id = {e["entity_id"]: e for e in entities}
    player_id = snapshot.get("perspective", {}).get("player_id")
    players = [e for e in entities if e.get("type") == "PLAYER"]
    me = next((e for e in players if player_id is not None and
               e.get("tags", {}).get("PLAYER_ID", e.get("controller")) == player_id), None)
    controller = me.get("controller") if me else None
    game = next((e for e in entities if e.get("type") == "GAME"), {})
    combat = game.get("tags", {}).get("BACON_IN_COMBAT_PHASE")
    shop_controllers, opponent_controllers = set(), set()
    for player in players:
        owner = player.get("controller")
        if owner is None or owner == controller:
            continue
        tags = player.get("tags", {})
        hero = by_id.get(tags.get("HERO_ENTITY"), {})
        if hero.get("zone") != "PLAY" or hero.get("controller") != owner:
            continue
        # The same dummy controller represents Bob in recruit and an opponent in combat.
        if hero.get("card_id") == "TB_BaconShopBob" and combat != 1:
            shop_controllers.add(owner)
        elif combat == 1 and hero.get("card_id") != "TB_BaconShopBob" and tags.get("BACON_CURRENT_COMBAT_PLAYER_ID", 0) > 0:
            opponent_controllers.add(owner)

    choice = snapshot.get("last_choices") or {}
    choice_ids = set(choice.get("entity_ids", [])) if (
        not choice.get("selection_sent_or_resolved", True) and me and
        choice.get("player_entity_id") == me["entity_id"]
    ) else set()
    cards, historical = [], 0
    for entity in entities:
        kind, zone, owner = entity.get("type"), entity.get("zone"), entity.get("controller")
        if kind in INTERNAL_TYPES or entity.get("catalog_category") == "internal":
            continue
        is_choice = entity["entity_id"] in choice_ids
        if not entity.get("card_id") and kind in (None, "INVALID") and not is_choice:
            continue  # Hidden/cleared entity references are not visible cards.
        if zone not in ("PLAY", "HAND") and not is_choice:
            if entity.get("card_id"):
                historical += 1
            continue
        if entity.get("card_id") == "TB_BaconShopBob":
            continue
        # Never list an opponent's known or hidden hand as a visible board card.
        if zone == "HAND" and (controller is None or owner != controller) and not is_choice:
            continue
        if is_choice:
            group = "choices"
        elif controller is not None and owner == controller:
            group = "hand" if zone == "HAND" else "board" if kind == "MINION" else "player"
        elif owner in shop_controllers and zone == "PLAY" and kind in ("MINION", "BATTLEGROUND_SPELL"):
            group = "shop"
        elif owner in opponent_controllers and zone == "PLAY":
            group = "opponent"
        else:
            group = "other"
        position, tags = entity.get("position"), entity.get("tags", {})
        detail = []
        if isinstance(position, int) and position > 0:
            detail.append(f"第 {position} 位")
        if kind == "MINION" and "ATK" in tags and "HEALTH" in tags:
            detail.append(f"{tags['ATK']}/{max(0, tags['HEALTH'] - tags.get('DAMAGE', 0))}")
        if tags.get("FROZEN"):
            detail.append("已冻结")
        if group == "other":
            detail.append(f"控制方 {owner}" if owner is not None else "缺少归属信息")
        if not detail:
            detail.append({"HERO": "英雄", "HERO_POWER": "英雄技能", "BATTLEGROUND_TRINKET": "饰品"}.get(kind, "日志可见卡牌"))
        name = entity.get("catalog_name_zh") or entity.get("catalog_name")
        if not name:
            name = "英雄皮肤 · 名称未映射" if entity.get("catalog_category") == "hero_skin" else "未知卡牌"
        cards.append({"id": str(entity["entity_id"]), "name": name,
            "detail": " · ".join(detail), "card_id": entity.get("card_id"),
            "group": group, "group_title": GROUPS[group], "position": position})
    order = {key: i for i, key in enumerate(GROUPS)}
    cards.sort(key=lambda row: (order[row["group"]], row["position"] or 0, int(row["id"])))
    phase = "战斗" if combat == 1 else "招募" if shop_controllers else "阶段切换中"
    if snapshot.get("game_status") == "complete":
        phase = "本局已结束"
    elif choice_ids and combat is None:
        phase = "选择中"
    perspective_source = "客户端读取" if snapshot.get("perspective", {}).get("source") == "hearthmirror" else "日志推断"
    facts = [{"label": "记录阶段", "value": phase},
             {"label": "玩家视角", "value": f"玩家 {player_id} · {perspective_source}" if me else "尚未确定"}]
    if me:
        tags = me.get("tags", {})
        if tags.get("PLAYER_TECH_LEVEL"):
            facts.append({"label": "酒馆等级", "value": str(tags["PLAYER_TECH_LEVEL"])})
        if "TURN" in tags:
            facts.append({"label": "回合", "value": str(tags["TURN"])})
    if historical:
        facts.append({"label": "已收起非当前实体", "value": str(historical)})
    return cards, facts
