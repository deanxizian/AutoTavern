"""Regressions extracted from the completed 2026-10-30 Faelin match.

Fixtures contain sanitized visible entities, not raw account identifiers. The
source digest identifies the full local recording used for the offline replay.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
import taverngym as tg
from tools.client_actions import bind_actions, frame_identity, match_submission
from tools.client_decisions import pending_observation
from tools.client_encoding import encode_public_view
from tools.client_features import ClientFeatureAdapter
from tools.client_model import PolicyRuntime, PolicyResult


@pytest.fixture(scope="module")
def recording():
    with gzip.open(Path(__file__).parent / "fixtures/client_match_20261030.json.gz", "rt") as stream:
        return json.load(stream)


@pytest.fixture(scope="module")
def adapter():
    return ClientFeatureAdapter()


def test_locked_hero_offers_keep_display_order_and_never_enter_action_mask(recording, adapter):
    packet = adapter.convert(recording["snapshots"]["heroes"])
    assert packet.complete
    assert packet.actions.phase == "hero_select"
    assert set(packet.actions.bindings) == {31, 32}
    assert packet.cards[44:48].tolist() == [adapter.vocabulary[c] for c in
        ("BG20_HERO_280", "BG21_HERO_020", "BG22_HERO_201", "TB_BaconShop_HERO_14")]
    assert packet.strategy_view["me"]["hand"] == []
    assert packet.features[11] == np.float32(.6)


@pytest.mark.parametrize("name", ["bellringer_move_first", "bellringer_target_first"])
def test_target_children_identify_activate_independent_of_packet_order(recording, name):
    snap = recording["snapshots"][name]
    mapping = bind_actions(snap)
    activate = next(b for b in mapping.bindings.values() if b["kind"] == "activate")
    entity = activate["source_entity_id"]
    assert not mapping.issues
    assert activate["targets"]
    moves = [b for b in mapping.bindings.values() if b["kind"] == "move" and b["source_entity_id"] == entity]
    assert moves and all(b["option_id"] != activate["option_id"] for b in moves)
    snap = deepcopy(snap)
    target_menu = next(o for o in snap["last_options"]["options"] if o["id"] == activate["option_id"])
    target_menu["error"] = "REQ_ENOUGH_MANA"
    mapping = bind_actions(snap)
    assert not any(b["kind"] == "activate" and b["source_entity_id"] == entity for b in mapping.bindings.values())
    assert any(b["kind"] == "move" and b["source_entity_id"] == entity for b in mapping.bindings.values())


def test_recorded_activate_and_nested_choose_one_submission_match(recording):
    for name, expected in (("bellringer_target_first", "requires_target_decision"), ("alliance", "requires_mode_decision")):
        a = bind_actions(recording["snapshots"][name])
        event = deepcopy(recording["events"][name])
        assert match_submission(a, event)["status"] == expected
        event["target"]["entity_id"] = 999999
        assert match_submission(a, event)["status"] == "unmapped"
    event = deepcopy(recording["events"]["alliance"])
    event["suboption"] = 9
    assert match_submission(a, event)["status"] == "unmapped"


@pytest.mark.parametrize("cancel_target", [False, True])
def test_choose_one_resolves_its_own_target_menu_or_rejects_the_opening_action(recording, adapter, cancel_target):
    packet = adapter.convert(recording["snapshots"]["alliance"])
    seen = []
    def infer(obs, *, strategy_view):
        pending = strategy_view["me"]["pending"] or {}
        seen.append(pending.get("kind"))
        if len(seen) == 1: action = 54
        elif len(seen) == 2: action = 31
        elif len(seen) == 3: action = 7 if cancel_target else 30 + pending["targets"].index(9446)
        else:
            assert not obs["action_mask"][0, 54]
            action = 0
        assert obs["action_mask"][0, action]
        return PolicyResult(action, .75, .1, 1.)
    runtime = SimpleNamespace(adapter=adapter, metadata={}, infer_arrays=infer)
    result = PolicyRuntime.infer_client(runtime, packet)
    assert seen[:3] == [None, "choose_one", "target"]
    if cancel_target:
        assert result["action"] == 0 and result["excluded_actions"]
    else:
        assert result["binding"]["resolved_mode"]["id"] == 1
        assert result["binding"]["resolved_target"]["entity_id"] == 9446
        assert "联盟盾牌" in result["instruction"] and "战场第 6 位" in result["instruction"]
        assert not result["excluded_actions"]
    # Preview must never commit payment, remove the source, or change the packet.
    assert packet.actions.bindings[54].get("resolved_mode") is None
    assert len(packet.strategy_view["me"]["hand"]) == 2


def test_faelin_locked_cards_and_max_tier_without_upgrade_button(recording, adapter):
    packet = adapter.convert(recording["snapshots"]["faelin"])
    assert packet.complete and packet.strategy_view["me"]["hand"]
    assert all(h["locked"] == 1 for h in packet.strategy_view["me"]["hand"])
    assert not any(b["kind"] == "play_minion" for b in packet.actions.bindings.values())
    snap = deepcopy(recording["snapshots"]["tier6"])
    packet = adapter.convert(snap)
    assert packet.complete and packet.features[11] == 0 and not packet.action_mask[3]
    own = next(e for e in snap["entities"] if e["type"] == "PLAYER" and e["tags"].get("PLAYER_ID") == 6)
    own["tags"]["PLAYER_TECH_LEVEL"] = 5
    assert adapter.convert(snap).gaps == [{"code": "button_missing:TB_BaconShopTechUp"}]


def test_choice_packet_completion_changes_frame_identity(recording):
    snap = deepcopy(recording["snapshots"]["heroes"])
    snap["choice_packet_complete"] = False
    snap["choices_after_last_state_change"] = False
    before = frame_identity(snap)
    assert {x["code"] for x in bind_actions(snap).issues} == {"stale_choices"}
    snap["choice_packet_complete"] = True
    assert frame_identity(snap) != before
    assert not bind_actions(snap).issues


def test_discover_inside_an_open_server_block_does_not_wait_for_its_own_answer(recording, adapter):
    snap = deepcopy(recording["snapshots"]["faelin"])
    assert snap["last_choices"] and not snap["last_choices"]["selection_sent_or_resolved"]
    snap.update(status="incomplete_snapshot", open_blocks=2, choice_packet_complete=True)
    assert adapter.convert(snap).complete
    snap["source"]["partial_trailing_line"] = True
    assert adapter.convert(snap).gaps == [{"code": "unstable_snapshot"}]
    snap["source"]["partial_trailing_line"] = False
    snap["choice_packet_complete"] = False
    assert adapter.convert(snap).gaps == [{"code": "stale_choices"}]


def test_encoder_matches_native_public_contract_across_a_complete_game(adapter):
    from training.attachments import AttachmentRows
    from training.growth import from_views
    game = tg.Game(739175, json.dumps({"hotfix_37_0_1": True, "turn_action_caps": [128]}))
    count, phases, kinds = 0, set(), set()
    for _ in range(1000):
        active = False
        for seat in range(8):
            mask = np.frombuffer(game.legal_mask(seat), dtype=np.bool_)
            if not mask.any(): continue
            active = True
            view = json.loads(game.strategy_view(seat))
            # Test oracle only: these two public counters are omitted by the
            # native JSON accessor. Never use serialized games for live input.
            raw = json.loads(game.state_json())
            player = raw["players"][seat]
            view["me"]["chronum_saved"] = player["chronum_saved"]
            for h, native in zip(view["me"]["hand"], player["hand"]):
                m = native["item"].get("Minion", {})
                h["doomed"] = bool(m.get("doomed_turn") and m["doomed_turn"] == raw["turn"])
            f, c, attachments, growth = encode_public_view(view, adapter.vocabulary)
            nf, nc = game.observe(seat)
            np.testing.assert_array_equal(f, np.frombuffer(nf, dtype="<f4"))
            np.testing.assert_array_equal(c, np.frombuffer(nc, dtype="<i4"))
            native = AttachmentRows.from_games([game], [(0, seat)])
            np.testing.assert_array_equal(attachments.counts, native.counts)
            np.testing.assert_array_equal(attachments.cards, native.cards)
            np.testing.assert_array_equal(growth, from_views([view]))
            count += 1
            phases.add(view["phase"])
            kinds.add((view["me"]["pending"] or {}).get("kind"))
            game.step(seat, game.bot_action(seat))
        if not active: break
    assert count >= 500 and phases == {"hero_select", "recruit"}
    assert {"hero", "discover", "target", "trinket", None} <= kinds
