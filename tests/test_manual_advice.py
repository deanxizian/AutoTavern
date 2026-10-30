"""Human-action ambiguity, visibility and resource costs, without a client."""

import copy
import json

import pytest

from tools.manual_advice import ACTION_LAYOUT, render_action, render_or_block


def minion(name, entity, *, kind="minion", attack=3, health=4):
    return {"id": "CARD_" + name, "name": name, "name_zh": name, "type": kind,
            "eid": entity, "sid": entity, "hid": entity, "atk": attack, "hp": health,
            "golden": False, "price": 3, "costs_health": False, "frozen": False,
            "activate_cost": 1}


def view(action):
    mask = [0] * 245
    mask[action] = 1
    return {"phase": "recruit", "actions": dict(ACTION_LAYOUT), "mask": mask,
            "me": {"alive": True, "gold": 5, "tier": 2, "upgrade_price": 4,
                   "board": [minion("甲", 101), minion("乙", 102), minion("丙", 103)],
                   "hand": [minion("手牌甲", 201), minion("手牌乙", 202)],
                   "shop": [minion("商品甲", 301), minion("商品乙", 302)],
                   "pending": None, "cancellable": True, "dark_gift": {"cost": 1},
                   "hero_powers": [{"id": "POWER", "name_zh": "瞄准",
                                    "slot": 0, "action": 4, "price": 0}]},
            "opponents": [{"board": [minion("PRIVATE_OPPONENT_CARD", 999)]}],
            "combats": [{"unrelated": "PRIVATE_COMBAT"}]}


def test_duplicate_names_still_identify_the_correct_slot_and_health_payment():
    v = view(9)
    v["me"]["shop"][0]["name_zh"] = v["me"]["shop"][1]["name_zh"] = "同名随从"
    v["me"]["shop"][1].update(costs_health=True, price=2, atk=9, golden=True)
    result = render_action(v, 9)
    assert "酒馆第 2 位「同名随从」（金色，9/4）" in result["instruction"]
    assert "2 生命值" in result["instruction"]
    assert result["references"][0]["entity_id"] == 302
    assert result["cost"] == {"amount": 2, "currency": "health"}
    assert not result["live_ready"]


def test_end_turn_does_not_invent_a_client_button_or_mutate_state():
    v = view(0)
    before = copy.deepcopy(v)
    result = render_action(v, 0)
    assert result["instruction"] == "本回合不再操作，等待战斗开始。"
    assert v == before and "PRIVATE" not in json.dumps(result)


def test_partial_freeze_freezes_all_and_only_full_freeze_unfreezes():
    v = view(2)
    v["me"]["shop"][0]["frozen"] = True
    assert render_action(v, 2)["kind"] == "freeze"
    v["me"]["shop"][1]["frozen"] = True
    assert render_action(v, 2)["kind"] == "unfreeze"
    v["me"]["shop"] = []
    assert render_action(v, 2)["kind"] == "freeze"


def test_play_uses_eight_position_slots_and_has_distinct_magnetic_instruction():
    v = view(46 + 8 + 2)
    result = render_action(v, 56)
    assert result["references"][0]["entity_id"] == 202
    assert result["references"][1]["entity_id"] == 103
    assert result["position_after"] == 3
    assert "单独打出" in result["instruction"] and "左侧" in result["instruction"]
    v["mask"][126 + 7 + 2] = 1
    magnetic = render_action(v, 135)
    assert magnetic["kind"] == "magnetize"
    assert [r["entity_id"] for r in magnetic["references"]] == [202, 103]
    assert "磁力合体" in magnetic["instruction"]


def test_right_edge_insertion_move_and_spell_have_unambiguous_positions():
    v = view(49)
    assert "丙」" in render_action(v, 49)["instruction"]
    assert "右侧" in render_action(v, 49)["instruction"]
    v["mask"][198] = 1
    move = render_action(v, 198)
    assert move["references"][0]["entity_id"] == 101 and move["position_after"] == 3
    v["mask"][46] = 1
    v["me"]["hand"][0]["type"] = "spell"
    assert render_action(v, 46)["kind"] == "cast"
    v["mask"][47] = 1
    assert render_or_block(v, 47)["reason"] == "unsupported_play"


@pytest.mark.parametrize("zone,key,target", [("board", "eid", 102), ("shop", "sid", 302),
                                             ("shop", "eid", 333), ("hand", "hid", 202)])
def test_targets_resolve_visible_entities_not_option_ordinal(zone, key, target):
    v = view(30)
    v["me"][zone][1][key] = target
    v["me"]["pending"] = {"kind": "target", "targets": [target]}
    result = render_action(v, 30)
    assert result["references"][0]["zone"] == zone
    assert result["references"][0]["position"] == 2
    assert result["target_entity_id"] == target


def test_hidden_duplicate_or_missing_targets_never_get_guessed():
    v = view(30)
    v["me"]["pending"] = {"kind": "target", "targets": [999]}
    assert render_or_block(v, 30)["reason"] == "unresolved_target"
    v["me"]["pending"]["targets"] = [101]
    v["me"]["shop"][0]["sid"] = 101
    assert render_or_block(v, 30)["reason"] == "unresolved_target"


def test_choose_one_placeholder_cards_do_not_become_fabricated_mode_labels():
    v = view(30)
    card = minion("抉择来源", 400)
    v["me"]["pending"] = {"kind": "choose_one", "source": card, "options": [card, card]}
    result = render_or_block(v, 30)
    assert result["status"] == "blocked" and result["instruction"] is None
    assert result["reason"] == "unresolved_choose_one"


def test_discover_hero_trinket_quest_and_timewarp_choices_keep_their_identity():
    v = view(31)
    options = [minion("选项甲", 401), minion("选项乙", 402)]
    for kind in ("hero", "hero_power", "discover"):
        v["me"]["pending"] = {"kind": kind, "options": copy.deepcopy(options)}
        assert render_action(v, 31)["references"][0]["name"] == "选项乙"
    v["me"]["pending"] = {"kind": "trinket", "free": True, "options": options}
    assert render_action(v, 31)["cost"]["amount"] == 0
    v["me"]["pending"]["free"] = False
    options[1]["cost"] = 6
    assert render_action(v, 31)["cost"]["amount"] == 6
    options[1]["reward"] = {"id": "REWARD", "name_zh": "任务奖励"}
    v["me"]["pending"]["kind"] = "quest"
    assert "任务奖励" in render_action(v, 31)["instruction"]
    options[1]["chronum_cost"] = 2
    v["me"]["pending"] = {"kind": "timewarp", "leave": 2, "options": options}
    assert render_action(v, 31)["cost"] == {"amount": 2, "currency": "chronum"}
    v["mask"][32] = 1
    assert render_action(v, 32)["kind"] == "leave_timewarp"


@pytest.mark.parametrize("action,kind", [(1, "refresh"), (3, "upgrade"), (4, "hero_power"),
                                        (6, "dark_gift"), (7, "cancel"), (17, "sell"),
                                        (24, "activate")])
def test_remaining_buttons_and_card_operations_are_specific(action, kind):
    result = render_action(view(action), action)
    assert result["kind"] == kind and result["instruction"]
    if action == 1:
        assert "cost" not in result  # Native refresh_price alone cannot identify currency.
    if action in (17, 24):
        assert result["references"][0]["entity_id"] == 102


@pytest.mark.parametrize("mutation,reason", [
    (lambda v: v["actions"].update(N_POS=7), "action_layout_mismatch"),
    (lambda v: v["mask"].__setitem__(8, 0), "illegal_action"),
    (lambda v: v["mask"].__setitem__(8, "1"), "invalid_mask"),
    (lambda v: v.update(phase="game_over"), "not_action_phase"),
    (lambda v: v["me"]["shop"][0].update(price=None), "incomplete_state"),
    (lambda v: v["me"]["shop"][0].update(id=None), "unknown_card"),
    (lambda v: v["me"].update(shop=[]), "missing_position"),
    (lambda v: v.update(model_ready=True), "client_adapter_unvalidated"),
])
def test_missing_or_stale_state_is_blocked_without_an_alternative_action(mutation, reason):
    v = view(8)
    mutation(v)
    result = render_or_block(v, 8)
    assert result["reason"] == reason and result["instruction"] is None
    assert "action" not in result and not result["live_ready"]


def test_diagnostic_client_snapshot_is_not_a_model_input():
    result = render_or_block({"status": "empty_log", "model_ready": False}, 0)
    assert result["reason"] == "client_adapter_unvalidated"


def test_preview_binds_acting_player_state_and_rejects_changed_order():
    v = view(8)
    first = render_action(v, 8)
    assert first["execution_recorded"] is False
    v["opponents"] = [{"hand": "PRIVATE_CHANGED"}]
    assert render_action(v, 8)["view_id"] == first["view_id"]
    v["me"]["shop"].reverse()
    result = render_or_block(v, 8, expected_view_id=first["view_id"])
    assert result["reason"] == "state_changed" and result["instruction"] is None


def test_typed_client_record_cannot_impersonate_simulator_by_removing_ready_flag():
    v = view(8)
    v["schema"] = "autotavern.power-log.v1"
    assert render_or_block(v, 8)["reason"] == "client_adapter_unvalidated"
