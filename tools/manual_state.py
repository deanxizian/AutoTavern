"""Collect and check visible manual records; never create policy inputs or actions."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import uuid

if __package__:
    from .client_common import DEFAULT_MAX_AGE, content_id, freshness, load_catalog, write_private_json
else:
    from client_common import DEFAULT_MAX_AGE, content_id, freshness, load_catalog, write_private_json


SCHEMA = "autotavern.manual-state.v1"
ZONES = ("board", "hand", "shop")
BLOCKERS = ["manual_to_model_mapping_unvalidated", "legal_action_mapping_unvalidated",
            "client_rules_unverified", "model_readiness_not_established"]


def template():
    # null means unknown; [] means the human checked that the list is empty.
    return {"schema": SCHEMA, "capture_id": uuid.uuid4().hex, "observed_at": None,
            "client_build": None, "confirmed": False, "phase": None, "turn": None,
            "hero": None, "resources": {"health": None, "armor": None, "gold": None,
                                        "tavern_tier": None},
            "zones": {zone: None for zone in ZONES}, "hero_powers": None,
            "visible_effects": None,
            "pending": {"kind": None, "options": None, "targets": None},
            "legal_actions": {"confirmed": False, "visible_labels": None}}


def validate_entry(entry, *, now=None, max_age=DEFAULT_MAX_AGE, expected_state_id=None):
    result = {"schema": "autotavern.manual-check.v1", "scope": "manual_record_validation",
              "status": "needs_input", "model_ready": False, "live_ready": False,
              "model_blockers": list(BLOCKERS), "issues": []}
    issues = result["issues"]
    def issue(path, code, message):
        issues.append({"path": path, "code": code, "message": message})
    def mapping(value, path, allowed):
        if not isinstance(value, dict):
            issue(path, "missing_fields", "请填写这一组可见信息；未知值保留 null。")
            return {}
        if set(value) - set(allowed):
            issue(path, "unsupported_fields", "包含格式之外的字段；不要录入对手隐藏信息或模拟器状态。")
        return value
    def items(value, path):
        if not isinstance(value, list):
            issue(path, "missing_list", "请按从左到右填写；已确认没有内容时填 []。")
            return []
        return value
    def integer(value, path, minimum=0):
        if type(value) is not int or value < minimum:
            issue(path, "missing_number", "请填写屏幕显示的非负整数，不要采用卡牌基础数值。")
    def labels(value, path, *, nonempty=False):
        rows = items(value, path)
        if (nonempty and not rows) or any(not isinstance(s, str) or not s.strip() for s in rows):
            issue(path, "missing_visible_labels", "请填写实际可见的完整标签，不能据此自动推导动作编号。")
    def cost(value, path):
        value = mapping(value, path, ("amount", "currency"))
        integer(value.get("amount"), path + ".amount")
        if value.get("currency") not in ("gold", "health", "chronum"):
            issue(path + ".currency", "unknown_currency", "请确认费用单位：gold、health 或 chronum。")

    if not isinstance(entry, dict) or entry.get("schema") != SCHEMA:
        issue("schema", "wrong_schema", "需要手动录入模板；不能使用 Power.log 诊断或 Game.view 代替。")
        return result
    mapping(entry, "$", template().keys())
    try:
        result["state_id"] = content_id(entry)
    except (ValueError, TypeError, OverflowError):
        issue("$", "invalid_json_values", "局面包含无效 JSON 数值。")
    if expected_state_id is not None and result.get("state_id") != expected_state_id:
        issue("state_id", "state_changed", "记录已改变，旧的核对结果必须作废并重新核对。")
    if not isinstance(entry.get("capture_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", entry["capture_id"]):
        issue("capture_id", "missing_capture_id", "每次读取后请使用新模板的 capture_id。")
    result["freshness"] = freshness(entry.get("observed_at"), now=now, max_age=max_age)
    result["freshness"]["basis"] = "human_supplied_time_not_live_client"
    if result["freshness"]["status"] != "recent":
        issue("observed_at", "record_" + result["freshness"]["status"],
              "填写实际观察时间（含时区）；旧记录不能作为当前局面。")
    build = entry.get("client_build")
    if not isinstance(build, str) or not re.fullmatch(r"\d+(?:\.\d+){2,3}", build):
        issue("client_build", "missing_client_build", "填写已核对的客户端构建号；这不等于服务端规则验证。")
    if entry.get("confirmed") is not True:
        issue("confirmed", "visible_state_unconfirmed", "逐项对照可见界面后才能将 confirmed 设为 true。")
    phase = entry.get("phase")
    if phase not in ("hero_select", "recruit", "combat", "game_over"):
        issue("phase", "unknown_phase", "确认阶段：hero_select、recruit、combat 或 game_over。")
    integer(entry.get("turn"), "turn")
    meta, catalog = load_catalog()
    result["catalog_client"] = meta["client_version"]
    if isinstance(build, str) and build != meta["client_version"]:
        result["model_blockers"].append("client_differs_from_catalog")
    seen_entities = set()

    def card(value, path, *, kind=None, zone=None):
        value = mapping(value, path, ("card_id", "entity_id", "attack", "health", "golden",
                                      "frozen", "cost", "visible_text", "label"))
        card_id = value.get("card_id")
        known = catalog.get(card_id) if isinstance(card_id, str) else None
        if not known:
            issue(path + ".card_id", "unknown_card", "卡牌 ID 缺失或不在目录；请用 lookup 核对，不得近似替换。")
        elif kind and known["kind"] != kind:
            issue(path + ".card_id", "card_kind_mismatch", "卡牌类型与所在位置不符，请重新核对 ID。")
        elif zone in ("hand", "shop") and known["kind"] not in ("minion", "spell", "tavern_spell"):
            issue(path + ".card_id", "card_kind_mismatch", "手牌和酒馆条目必须是随从或法术。")
        if "entity_id" in value:
            entity = value["entity_id"]
            integer(entity, path + ".entity_id", 1)
            if type(entity) is int and entity > 0:
                # Option cards may refer to a board entity; only physical zones are unique.
                if zone in ZONES:
                    if entity in seen_entities:
                        issue(path + ".entity_id", "duplicate_entity", "不同区域重复使用了同一个实体 ID。")
                    seen_entities.add(entity)
        if known and known["kind"] == "minion":
            integer(value.get("attack"), path + ".attack")
            integer(value.get("health"), path + ".health")
            if type(value.get("golden")) is not bool:
                issue(path + ".golden", "unknown_golden", "请明确标记是否为金色随从。")
            elif value["golden"] != bool(known.get("normal")):
                issue(path + ".golden", "golden_id_mismatch", "金色状态与目录 ID 不一致；需核对对应金色 ID。")
        if zone == "shop":
            if type(value.get("frozen")) is not bool:
                issue(path + ".frozen", "unknown_frozen", "请确认这张商品是否被冻结。")
            cost(value.get("cost"), path + ".cost")
        elif "cost" in value:
            cost(value["cost"], path + ".cost")
        for field in ("visible_text", "label"):
            if field in value and (not isinstance(value[field], str) or not value[field].strip()):
                issue(path + "." + field, "missing_visible_label", "请填写实际显示的文字。")

    if phase != "hero_select" or entry.get("hero") is not None:
        card(entry.get("hero"), "hero", kind="hero")
    resources = mapping(entry.get("resources"), "resources", ("health", "armor", "gold", "tavern_tier"))
    for key in ("health", "armor", "gold", "tavern_tier"):
        if phase != "hero_select" or resources.get(key) is not None:
            integer(resources.get(key), "resources." + key, 1 if key == "tavern_tier" else 0)
    zones = mapping(entry.get("zones"), "zones", ZONES)
    for zone in ZONES:
        for index, row in enumerate(items(zones.get(zone), "zones." + zone)):
            card(row, f"zones.{zone}[{index}]", kind="minion" if zone == "board" else None, zone=zone)
    for index, row in enumerate(items(entry.get("hero_powers"), "hero_powers")):
        card(row, f"hero_powers[{index}]", kind="hero_power")
        if isinstance(row, dict):
            cost(row.get("cost"), f"hero_powers[{index}].cost")
    labels(entry.get("visible_effects"), "visible_effects")
    pending = mapping(entry.get("pending"), "pending", ("kind", "options", "targets"))
    kind = pending.get("kind")
    if kind not in ("none", "hero", "hero_power", "discover", "target", "choose_one",
                    "trinket", "quest", "timewarp"):
        issue("pending.kind", "unknown_pending", "确认当前选择类型；无选择窗口时填 none。")
    options = items(pending.get("options"), "pending.options")
    targets = items(pending.get("targets"), "pending.targets")
    if kind == "none" and (options or targets):
        issue("pending", "inconsistent_pending", "无选择窗口时选项和目标都应为空。")
    if kind == "target" and (not targets or options):
        issue("pending.targets", "missing_targets", "目标选择需要当前可见目标，不能混用卡牌选择列表。")
    if kind not in ("none", "target") and (not options or targets):
        issue("pending.options", "missing_choices", "请按顺序填写实际选项；目标列表应为空。")
    if phase == "hero_select" and kind != "hero":
        issue("pending.kind", "missing_hero_choices", "选英雄阶段需要实际英雄选项。")
    for index, row in enumerate(options):
        card(row, f"pending.options[{index}]", kind="hero" if kind == "hero" else None)
        if kind == "choose_one" and (not isinstance(row, dict) or not row.get("label")):
            issue(f"pending.options[{index}].label", "unresolved_choose_one", "填写实际抉择模式标签，不能复制来源卡牌名。")
    if kind == "choose_one":
        mode_labels = [row.get("label") for row in options if isinstance(row, dict)]
        if len(options) != 2 or len(mode_labels) != 2 or mode_labels[0] == mode_labels[1]:
            issue("pending.options", "unresolved_choose_one", "需要两个不同的实际抉择模式，不能复用来源卡牌名。")
    for index, target in enumerate(targets):
        path = f"pending.targets[{index}]"
        target = mapping(target, path, ("zone", "position"))
        zone, position = target.get("zone"), target.get("position")
        zone_rows = zones.get(zone) if isinstance(zone, str) and zone in ZONES else None
        if not isinstance(zone_rows, list) or type(position) is not int or not 1 <= position <= len(zone_rows):
            issue(path, "unresolved_target", "目标必须指向可见 board/hand/shop 中从 1 开始的位置。")
    legal = mapping(entry.get("legal_actions"), "legal_actions", ("confirmed", "visible_labels"))
    if legal.get("confirmed") is not True:
        issue("legal_actions.confirmed", "missing_legal_actions", "逐项确认当前可用操作；不能沿用上一局面的列表。")
    labels(legal.get("visible_labels"), "legal_actions.visible_labels",
           nonempty=phase in ("hero_select", "recruit"))
    if not issues:
        result["status"] = "record_complete"
    result["model_blockers"] += sorted({i["code"] for i in issues})
    return result


def check_transition(before, after, **kwargs):
    """Compare two human captures; a change does not prove an action was executed."""
    result = validate_entry(after, **kwargs)
    transition = {"execution_verified": False, "changed_fields": []}
    result["transition"] = transition
    code = None
    if not isinstance(after, dict) or after.get("schema") != SCHEMA:
        code = "invalid_current_record"
    elif not isinstance(before, dict) or before.get("schema") != SCHEMA:
        code = "invalid_previous_record"
    elif before.get("capture_id") == after.get("capture_id"):
        code = "capture_reused"
    else:
        previous_check = validate_entry(before, now=kwargs.get("now"),
                                        max_age=kwargs.get("max_age", DEFAULT_MAX_AGE))
        if any(i["code"] != "record_stale" for i in previous_check["issues"]):
            code = "invalid_previous_record"
        try:
            old_time = datetime.fromisoformat(before["observed_at"])
            new_time = datetime.fromisoformat(after["observed_at"])
            if old_time.tzinfo is None or new_time.tzinfo is None or new_time <= old_time:
                code = "capture_time_not_advanced"
        except (KeyError, TypeError, ValueError):
            code = "capture_time_unknown"
        fields = ("client_build", "phase", "turn", "hero", "resources", "zones", "hero_powers",
                  "visible_effects", "pending", "legal_actions")
        transition["changed_fields"] = [field for field in fields if before.get(field) != after.get(field)]
        if not transition["changed_fields"]:
            code = code or "state_change_not_observed"
    if code:
        result["status"] = "needs_input"
        result["issues"].append({"path": "transition", "code": code,
                                 "message": "尚未确认操作后的新局面；不能据此认定操作已执行或继续建议。"})
        result["model_blockers"].append(code)
    return result


def lookup(query, limit=20):
    meta, cards = load_catalog()
    query = query.strip().casefold()
    exact, partial = [], []
    for card in cards.values():
        fields = [str(card.get(k, "")).casefold() for k in ("id", "name", "name_zh")]
        if query and any(query in f for f in fields):
            row = {k: card.get(k) for k in ("id", "name", "name_zh", "kind", "normal", "golden")}
            (exact if query in fields else partial).append(row)
    matches = exact + partial
    return {"catalog_version": meta["client_version"], "matches": matches[:limit],
            "total_matches": len(matches), "live_support_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("template", help="Create a new blank entry, never overwrite")
    create.add_argument("--out", type=Path, required=True)
    search = sub.add_parser("lookup", help="Find catalog IDs without guessing an identity")
    search.add_argument("query")
    check = sub.add_parser("check", help="List missing fields and freshness issues")
    check.add_argument("entry", type=Path)
    check.add_argument("--out", type=Path)
    check.add_argument("--previous", type=Path)
    check.add_argument("--expected-state-id")
    check.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE)
    args = parser.parse_args()
    if args.command == "check" and (not math.isfinite(args.max_age) or args.max_age <= 0):
        parser.error("--max-age must be finite and positive")
    try:
        if args.command == "template":
            write_private_json(args.out, template(), overwrite=False)
            result = {"status": "template_created", "out": str(args.out), "model_ready": False}
        elif args.command == "lookup":
            result = lookup(args.query)
        else:
            if args.out and args.out.resolve() in {p.resolve() for p in (args.entry, args.previous) if p}:
                parser.error("--out must not overwrite an input record")
            entry = json.loads(args.entry.read_text(encoding="utf-8"))
            options = {"max_age": args.max_age, "expected_state_id": args.expected_state_id}
            if args.previous:
                before = json.loads(args.previous.read_text(encoding="utf-8"))
                result = check_transition(before, entry, **options)
            else:
                result = validate_entry(entry, **options)
            if args.out:
                write_private_json(args.out, result)
    except (OSError, UnicodeError, ValueError) as exc:
        result = {"status": "blocked", "model_ready": False, "live_ready": False,
                  "reason": "output_exists" if isinstance(exc, FileExistsError) else "file_error",
                  "message": "未能读取或保存记录；模板不会覆盖已有文件。"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") in ("needs_input", "blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
