"""Render one simulator action as a Chinese instruction; never control a client.

This is an instruction renderer, not a live-state adapter or a policy. Its input
is the simulator's own view at the decision point. In particular, a diagnostic
Power.log snapshot is not an accepted input. No opponent data is copied out of
Game.view, whose debug form contains private current boards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from .client_common import content_id
else:
    from client_common import content_id


ACTION_LAYOUT = {
    "END": 0, "REFRESH": 1, "FREEZE": 2, "UPGRADE": 3, "HERO_POWER": 4,
    "DARK_GIFT": 6, "CANCEL": 7, "BUY": 8, "SELL": 16, "ACTIVATE": 23,
    "CHOOSE": 30, "PLAY": 46, "MAGNETIZE": 126, "MOVE": 196,
    "N_POS": 8, "N_BOARD": 7,
}
N_ACTIONS = 245
ZONE_NAMES = {"board": "我方战场", "hand": "手牌", "shop": "酒馆",
              "option": "选项", "power": "英雄技能", "reward": "奖励"}


class AdviceUnavailable(ValueError):
    """Missing or ambiguous information, with no raw state in the error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _fail(code, message):
    raise AdviceUnavailable(code, message)


def _integer(value, *, minimum=0):
    if type(value) is not int or value < minimum:
        _fail("incomplete_state", "局面缺少有效的数值，暂不生成指令。")
    return value


def _boolean(value):
    if type(value) is not bool:
        _fail("incomplete_state", "局面缺少明确的状态标记，暂不生成指令。")
    return value


def _mapping(value):
    if not isinstance(value, dict):
        _fail("incomplete_state", "局面缺少操作所需的信息，暂不生成指令。")
    return value


def _list(value):
    if not isinstance(value, list):
        _fail("incomplete_state", "局面缺少有序的卡牌或选项，暂不生成指令。")
    return value


def _at(items, index):
    items = _list(items)
    if not 0 <= index < len(items):
        _fail("missing_position", "指定位置不存在，需重新读取局面。")
    return _mapping(items[index])


def _reference(card, zone, index):
    card = _mapping(card)
    name = card.get("name_zh") or card.get("name")
    if not isinstance(name, str) or not name.strip() or not isinstance(card.get("id"), str) \
            or not card["id"].strip():
        _fail("unknown_card", "无法识别操作涉及的卡牌，暂不生成指令。")
    ref = {"zone": zone, "position": index + 1, "card_id": card["id"], "name": name}
    # Only zone-appropriate identity fields, never the original card/view object.
    identity = {"board": "eid", "hand": "hid", "shop": "sid"}.get(zone)
    if identity:
        ref["entity_id"] = _integer(card.get(identity), minimum=1)
    if card.get("type", card.get("kind")) == "minion" or zone == "board":
        ref.update(attack=_integer(card.get("atk")), health=_integer(card.get("hp")),
                   golden=_boolean(card.get("golden", bool(card.get("normal_id")))))
    return ref


def _label(ref):
    label = f"{ZONE_NAMES[ref['zone']]}第 {ref['position']} 位「{ref['name']}」"
    if "attack" in ref:
        label += f"（{'金色，' if ref['golden'] else ''}{ref['attack']}/{ref['health']}）"
    return label


def _cost(value, currency="gold"):
    amount = _integer(value)
    unit = {"gold": "金币", "health": "生命值", "chronum": "时空币"}[currency]
    return {"amount": amount, "currency": currency}, f"费用 {amount} {unit}"


def _target(me, entity):
    entity = _integer(entity, minimum=1)
    found = []
    for zone, keys in (("board", ("eid",)), ("shop", ("sid", "eid")), ("hand", ("hid",))):
        for index, card in enumerate(_list(me.get(zone))):
            card = _mapping(card)
            if any(card.get(key) == entity for key in keys):
                found.append(_reference(card, zone, index))
    if len(found) != 1:
        _fail("unresolved_target", "无法在自己的可见局面中唯一定位目标，暂不生成指令。")
    return found[0]


def _choose(me, option):
    pending = _mapping(me.get("pending"))
    kind = pending.get("kind")
    if kind == "target":
        targets = _list(pending.get("targets"))
        if not 0 <= option < len(targets):
            _fail("missing_position", "指定目标选项不存在，需重新读取局面。")
        ref = _target(me, targets[option])
        return {"kind": "target", "instruction": f"选择{_label(ref)}作为目标。",
                "references": [ref], "target_entity_id": targets[option]}
    if kind == "choose_one":
        # Native Game.view repeats the source card twice, not the two mode labels.
        _fail("unresolved_choose_one", "当前视图没有两个抉择模式的实际标签，暂不生成指令。")
    if kind not in ("hero", "discover", "hero_power", "trinket", "timewarp", "quest"):
        _fail("unsupported_choice", "当前选择类型尚未完成操作映射。")
    options = _list(pending.get("options"))
    if kind == "timewarp":
        leave = _integer(pending.get("leave"))
        if leave != len(options):
            _fail("incomplete_state", "离开酒馆的选项位置不一致，需重新读取局面。")
        if option == leave:
            return {"kind": "leave_timewarp", "instruction": "离开时空扭曲酒馆。",
                    "references": []}
    card = _at(options, option)
    ref = _reference(card, "option", option)
    result = {"kind": "choose_" + kind, "references": [ref], "option": option + 1}
    if kind == "trinket":
        cost, text = _cost(0 if _boolean(pending.get("free")) else card.get("cost"))
        result.update(cost=cost, instruction=f"选择{_label(ref)}（{text}）。")
    elif kind == "timewarp":
        cost, text = _cost(card.get("chronum_cost"), "chronum")
        result.update(cost=cost, instruction=f"购买{_label(ref)}（{text}）。")
    elif kind == "quest":
        reward = _reference(card.get("reward"), "reward", option)
        result["references"].append(reward)
        result["instruction"] = f"选择{_label(ref)}，对应奖励为「{reward['name']}」。"
    else:
        result["instruction"] = f"选择{_label(ref)}。"
    return result


def view_id(view):
    """Bind previews to acting-player fields; never hash/export private opponents."""
    try:
        return content_id({key: view.get(key) for key in ("phase", "turn", "seat", "version",
                                                        "actions", "mask", "me")})
    except (ValueError, TypeError, OverflowError):
        _fail("incomplete_state", "局面含有无效 JSON 数值。")


def render_action(view: dict, action: int, *, expected_view_id=None) -> dict:
    """Render an already selected legal native action; do not advance any state.

    The returned references have 1-based positions. This routine neither supplies
    model inputs nor validates that a simulator view matches the live client.
    """
    view = _mapping(view)
    if "model_ready" in view or "schema" in view:
        _fail("client_adapter_unvalidated", "客户端诊断快照尚未通过模型输入验证。")
    recording_id = view_id(view)
    if expected_view_id is not None and recording_id != expected_view_id:
        _fail("state_changed", "局面已变化，旧的动作预览必须作废并重新读取。")
    if view.get("actions") != ACTION_LAYOUT:
        _fail("action_layout_mismatch", "动作布局与指令转换器不一致。")
    if type(action) is not int or not 0 <= action < N_ACTIONS:
        _fail("invalid_action", "动作编号无效。")
    mask = _list(view.get("mask"))
    if len(mask) != N_ACTIONS or any(type(x) not in (bool, int) or x not in (0, 1) for x in mask):
        _fail("invalid_mask", "合法动作列表缺失或无效。")
    if not mask[action]:
        _fail("illegal_action", "该操作在当前局面不可用，需重新读取局面。")
    if view.get("phase") not in ("hero_select", "recruit"):
        _fail("not_action_phase", "当前不是可以操作的阶段。")
    me = _mapping(view.get("me"))
    if me.get("alive") is not True:
        _fail("not_action_phase", "当前玩家无法继续操作。")
    result = {"kind": "", "references": []}
    if action == 0:
        result.update(kind="wait_for_combat", instruction="本回合不再操作，等待战斗开始。")
    elif action == 1:
        # refresh_price alone does not say whether Malchezaar charges Health.
        # Do not mislabel it as Gold or duplicate native pricing rules here.
        result.update(kind="refresh", instruction="刷新一次酒馆。")
    elif action == 2:
        shop = _list(me.get("shop"))
        frozen = [_boolean(_mapping(card).get("frozen")) for card in shop]
        unfreeze = bool(frozen) and all(frozen)
        result.update(kind="unfreeze" if unfreeze else "freeze",
                      instruction="解冻酒馆中的卡牌。" if unfreeze else "冻结酒馆中的卡牌。")
    elif action == 3:
        tier = _integer(me.get("tier"), minimum=1)
        cost, text = _cost(me.get("upgrade_price"))
        result.update(kind="upgrade", cost=cost,
                      instruction=f"将酒馆从 {tier} 星升级到 {tier + 1} 星（{text}）。")
    elif 4 <= action < 6:
        slot = action - 4
        power = _at(me.get("hero_powers"), slot)
        if power.get("slot") != slot or power.get("action") != action:
            _fail("missing_position", "英雄技能的位置不一致，需重新读取局面。")
        ref = _reference(power, "power", slot)
        cost, text = _cost(power.get("price"))
        result.update(kind="hero_power", references=[ref], cost=cost,
                      instruction=f"使用{_label(ref)}（{text}）。")
    elif action == 6:
        cost, text = _cost(_mapping(me.get("dark_gift")).get("cost"))
        result.update(kind="dark_gift", cost=cost, instruction=f"使用黑暗之赐（{text}）。")
    elif action == 7:
        if me.get("cancellable") is not True:
            _fail("incomplete_state", "当前选择没有明确标记为可取消。")
        result.update(kind="cancel", instruction="取消当前的目标或模式选择。")
    elif action < 16:
        slot = action - 8
        card = _at(me.get("shop"), slot)
        ref = _reference(card, "shop", slot)
        currency = "health" if _boolean(card.get("costs_health")) else "gold"
        cost, text = _cost(card.get("price"), currency)
        result.update(kind="buy", references=[ref], cost=cost,
                      instruction=f"购买{_label(ref)}（{text}）。")
    elif action < 30:
        activate = action >= 23
        slot = action - (23 if activate else 16)
        card = _at(me.get("board"), slot)
        ref = _reference(card, "board", slot)
        if activate:
            cost, text = _cost(card.get("activate_cost"))
            result.update(kind="activate", references=[ref], cost=cost,
                          instruction=f"激活{_label(ref)}的主动效果（{text}）。")
        else:
            result.update(kind="sell", references=[ref], instruction=f"出售{_label(ref)}。")
    elif action < 46:
        result = _choose(me, action - 30)
    elif action < 126:
        hand, pos = divmod(action - 46, 8)
        card = _at(me.get("hand"), hand)
        ref = _reference(card, "hand", hand)
        result["references"] = [ref]
        if card.get("type") == "spell" and pos == 0:
            result.update(kind="cast", instruction=f"打出{_label(ref)}。")
        elif card.get("type") == "minion":
            board = _list(me.get("board"))
            if pos > len(board) or len(board) >= 7:
                _fail("missing_position", "随从放置位置无效，需重新读取局面。")
            if not board:
                destination = "我方空战场"
            else:
                neighbor_index = pos if pos < len(board) else len(board) - 1
                neighbor = _reference(_at(board, neighbor_index), "board", neighbor_index)
                result["references"].append(neighbor)
                destination = _label(neighbor) + ("的左侧" if pos < len(board) else "的右侧")
            result.update(kind="play_minion", position_after=pos + 1,
                          instruction=f"将{_label(ref)}单独打出到{destination}。")
        else:
            _fail("unsupported_play", "卡牌类型或法术放置位置尚未确认。")
    elif action < 196:
        hand, board = divmod(action - 126, 7)
        source = _reference(_at(me.get("hand"), hand), "hand", hand)
        target = _reference(_at(me.get("board"), board), "board", board)
        result.update(kind="magnetize", references=[source, target],
                      instruction=f"将{_label(source)}磁力合体到{_label(target)}。")
    else:
        source, dest = divmod(action - 196, 7)
        ref = _reference(_at(me.get("board"), source), "board", source)
        _at(me.get("board"), dest)
        if source == dest:
            _fail("invalid_action", "移动前后位置相同。")
        result.update(kind="move", references=[ref], position_after=dest + 1,
                      instruction=f"将{_label(ref)}移到我方战场第 {dest + 1} 位（从左到右计数）。")
    return {"status": "rendered", "scope": "simulator_action_preview", "live_ready": False,
            "view_id": recording_id, "execution_recorded": False, "action": action, **result}


def render_or_block(view, action, *, expected_view_id=None):
    """A JSON-safe result, with no invented fallback action on failure."""
    try:
        return render_action(view, action, expected_view_id=expected_view_id)
    except AdviceUnavailable as exc:
        return {"status": "blocked", "scope": "simulator_action_preview", "live_ready": False,
                "instruction": None, "reason": exc.code, "message": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True, help="Saved simulator Game.view JSON")
    parser.add_argument("--action", type=int, required=True, help="An already selected action index")
    parser.add_argument("--expected-view-id", help="Reject a saved view changed since a previous preview")
    args = parser.parse_args()
    try:
        view = json.loads(args.view.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        result = {"status": "blocked", "live_ready": False, "instruction": None,
                  "reason": "unreadable_view", "message": "无法读取模拟器局面文件。"}
    else:
        result = render_or_block(view, args.action, expected_view_id=args.expected_view_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "rendered" else 2


if __name__ == "__main__":
    raise SystemExit(main())
