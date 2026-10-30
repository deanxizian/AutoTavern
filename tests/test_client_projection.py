"""Ownership display must distinguish recruit shops from the combat dummy player."""

from copy import deepcopy

import pytest

from tools.client_projection import project_live_cards
from tools.window_session import WindowSession


def entity(eid, kind, owner=None, *, zone="PLAY", position=None, **tags):
    return {"entity_id": eid, "type": kind, "controller": owner, "zone": zone,
            "position": position, "card_id": f"card-{eid}", "catalog_name_zh": f"牌 {eid}", "tags": tags}


@pytest.fixture
def recruit():
    bob = entity(74, "HERO", 14)
    bob["card_id"] = "TB_BaconShopBob"
    return {"status": "snapshot", "game_status": "running", "perspective": {"player_id": 6},
            "entities": [entity(16, "GAME", BACON_IN_COMBAT_PHASE=0),
                entity(17, "PLAYER", 6, PLAYER_ID=6, HERO_ENTITY=115),
                entity(18, "PLAYER", 14, PLAYER_ID=14, HERO_ENTITY=74, BACON_DUMMY_PLAYER=1),
                bob, entity(115, "HERO", 6), entity(120, "MINION", 6, position=2, ATK=3, HEALTH=5, DAMAGE=1),
                entity(121, "MINION", 6, position=1), entity(122, "SPELL", 6, zone="HAND", position=1),
                entity(123, "MINION", 14, position=2, FROZEN=1), entity(124, "BATTLEGROUND_SPELL", 14, position=1),
                entity(125, "MINION", 6, zone="SETASIDE"), entity(126, "MINION", 14, zone="HAND"),
                entity(127, "ENCHANTMENT", 6)]}


def groups(snapshot):
    return {row["id"]: row["group"] for row in project_live_cards(snapshot)[0]}


def test_recruit_current_cards_are_sorted_and_history_internal_and_opponent_hand_are_excluded(recruit):
    cards, facts = project_live_cards(recruit)
    assert [row["id"] for row in cards] == ["121", "120", "122", "124", "123", "115"]
    assert groups(recruit) == {"121": "board", "120": "board", "122": "hand", "124": "shop", "123": "shop", "115": "player"}
    assert "3/4" in next(row["detail"] for row in cards if row["id"] == "120")
    assert "已冻结" in next(row["detail"] for row in cards if row["id"] == "123")
    assert {row["label"]: row["value"] for row in facts}["记录阶段"] == "招募"


def test_same_controller_becomes_combat_opponent_without_mislabeling_bob_as_shop(recruit):
    recruit["entities"][0]["tags"]["BACON_IN_COMBAT_PHASE"] = 1
    recruit["entities"][2]["tags"].update(HERO_ENTITY=201, BACON_CURRENT_COMBAT_PLAYER_ID=7)
    recruit["entities"].append(entity(201, "HERO", 14))
    classified = groups(recruit)
    assert classified["123"] == classified["201"] == "opponent"
    assert "shop" not in classified.values()


@pytest.mark.parametrize("phase", [None, 0, 1])
def test_ambiguous_or_transitioning_enemy_is_not_invented_as_shop(recruit, phase):
    recruit["entities"][0]["tags"]["BACON_IN_COMBAT_PHASE"] = phase
    recruit["entities"][2]["tags"]["HERO_ENTITY"] = 201
    recruit["entities"].append(entity(201, "HERO", 14))
    assert groups(recruit)["123"] == "other"


def test_missing_perspective_never_guesses_own_board_or_exposes_a_hand(recruit):
    recruit["perspective"] = {}
    classified = groups(recruit)
    assert classified["120"] == "other"
    assert "122" not in classified and "126" not in classified


def test_unresolved_own_choice_remains_visible_but_resolved_history_does_not(recruit):
    recruit["last_choices"] = {"player_entity_id": 17, "entity_ids": [125], "selection_sent_or_resolved": False}
    assert groups(recruit)["125"] == "choices"
    recruit["last_choices"]["selection_sent_or_resolved"] = True
    assert "125" not in groups(recruit)


def test_other_players_choices_do_not_enter_own_choice_group(recruit):
    recruit["last_choices"] = {"player_entity_id": 18, "entity_ids": [125], "selection_sent_or_resolved": False}
    assert "125" not in groups(recruit)


def test_unknown_card_stays_visible_and_projection_does_not_mutate_or_enable_advice(recruit):
    recruit["entities"][-5]["card_id"] = "unknown-new-card"
    recruit["entities"][-5]["catalog_name_zh"] = None
    before = deepcopy(recruit)
    class Feed:
        def sample(self, force=False):
            return {"snapshot": recruit, "running": True}
    state = WindowSession(Feed()).state()
    assert any(row["card_id"] == "unknown-new-card" and row["name"] == "未知卡牌" for row in state["cards"])
    assert state["model_ready"] is False and state["live_ready"] is False
    assert recruit == before


def test_history_cannot_fill_window_limit_and_hide_current_board(recruit):
    recruit["entities"] = [entity(2000+i, "MINION", 6, zone="SETASIDE") for i in range(100)] + recruit["entities"]
    assert "120" in groups(recruit) and len(groups(recruit)) == 6


def test_hidden_cleared_references_do_not_reappear_as_visible_opponents(recruit):
    cleared = entity(300, "INVALID", 14)
    cleared["card_id"] = None
    recruit["entities"].append(cleared)
    assert "300" not in groups(recruit)
