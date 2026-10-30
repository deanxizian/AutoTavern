"""Continuation contract checked against the unmodified training engine."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import taverngym as tg

from tools.client_features import ClientFeatureAdapter
from tools.client_decisions import pending_observation
from tools.client_encoding import encode_public_view
from tools.client_native import request


def public(game):
    v = json.loads(game.strategy_view(0))
    state = json.loads(game.state_json())
    v["me"]["chronum_saved"] = state["players"][0]["chronum_saved"]
    for h in v["me"]["hand"]: h["doomed"] = False
    return v


@pytest.mark.parametrize("cid,golden,choose", [("BG32_340", False, False), ("BG32_340", True, False),
    ("BG32_237", False, True), ("BG27_084", False, True)])
def test_post_summon_observation_matches_native_step(cid, golden, choose):
    game = tg.Game(127, json.dumps({"hotfix_37_0_1": True, "aberration_forced": False,
        "lobby_types": ["BEAST", "DRAGON", "ELEMENTAL", "MURLOC", "UNDEAD"]}))
    for seat in range(8): game.step(seat, 30)
    def debug(cmd, **kw):
        assert json.loads(game.debug(json.dumps({"cmd": cmd, "p": 0, **kw})))["ok"]
    debug("set_hero", hero="TB_BaconShop_HERO_21")
    for zone in ("board", "hand", "shop"): debug("clear_zone", zone=zone)
    debug("add_minion", zone="board", card="BG25_001", atk=11, hp=13)
    debug("add_minion", zone="board", card="BG27_084", atk=4, hp=7)
    debug("add_minion", zone="hand", card=cid, golden=golden)
    debug("edit_minion", zone="board", index=0, attach="BG36_MidGameEffect_000t22")
    # Test-only oracle: establish a valid already-applied global Undead aura.
    raw = json.loads(game.state_json())
    player = raw["players"][0]
    player["stats"]["undead_atk"] = 5
    for m in player["board"]+[h["item"]["Minion"] for h in player["hand"]]:
        if m["races"] & (1 << 8):
            m["atk"] += 5
            m["pbonus_atk"] += 5
    game = tg.Game.from_json(json.dumps(raw))
    v = public(game)
    source = v["me"]["hand"][0]
    native = game.copy()
    native.step(0, 47)  # Hand 0, board position 1 (zero based).
    expected = public(native)
    binding = {"kind": "play_minion", "source_entity_id": source["hid"], "position": 2,
        "targets": (expected["me"]["pending"] or {}).get("targets", [])}
    if choose:
        binding["modes"] = [{"id": 0}, {"id": 1}]
    ad = ClientFeatureAdapter()
    packet = SimpleNamespace(strategy_view=v)
    before = deepcopy(v)
    obs, projected, _ = pending_observation(packet, binding, ad.vocabulary)
    nf, nc = native.observe(0)
    np.testing.assert_array_equal(obs["features"][0], np.frombuffer(nf, dtype="<f4"))
    np.testing.assert_array_equal(obs["cards"][0], np.frombuffer(nc, dtype="<i4"))
    assert not obs["action_mask"][0, 7]  # Committed battlecry cannot cancel.
    assert projected["opponents"] == before["opponents"] and v == before
    if cid == "BG27_084":
        native.step(0, 31)
        expected = public(native)
        binding.pop("modes")
        binding.update(resolved_mode={"id": 1}, targets=expected["me"]["pending"]["targets"])
        obs, _, _ = pending_observation(packet, binding, ad.vocabulary)
        nf, nc = native.observe(0)
        np.testing.assert_array_equal(obs["features"][0], np.frombuffer(nf, dtype="<f4"))
        np.testing.assert_array_equal(obs["cards"][0], np.frombuffer(nc, dtype="<i4"))


def test_projection_preserves_paired_discard_and_player_auras():
    fixture = json.loads(gzip.decompress((Path(__file__).parent/"fixtures/client_match_20261030.json.gz").read_bytes()))
    ad = ClientFeatureAdapter()
    v = deepcopy(ad.convert(fixture["snapshots"]["faelin"]).strategy_view)
    h = v["me"]["hand"]
    h[0]["locked"] = 0
    h[0]["link"] = h[1]["hid"]
    h[0]["held"] = False
    source, paired = h[0]["hid"], h[1]["hid"]
    result = request(op="play", view=v, binding={"source_entity_id": source, "position": 1})
    assert all(c["hid"] not in (source, paired) for c in result["me"]["hand"])
    assert len(v["me"]["hand"]) == len(h)  # No mutation of observed state.


def test_special_gems_and_secondary_races_are_explicit_public_fields():
    fixture = json.loads(gzip.decompress((Path(__file__).parent/"fixtures/client_match_20261030.json.gz").read_bytes()))
    snap = deepcopy(fixture["snapshots"]["alliance"])
    ad = ClientFeatureAdapter()
    p = ad.convert(snap)
    hand = p.actions.zones["hand"][0]
    hand["card_id"], hand["type"] = "BG20_GEM_DivineShield", "SPELL"
    hand["tags"]["TAG_SCRIPT_DATA_NUM_1"] = 37  # Displayed stats, not variant.
    board = p.actions.zones["board"][0]
    board["tags"]["RACE_ALSO_NAGA"] = 1
    p = ad.convert(snap)
    assert p.complete
    assert p.strategy_view["me"]["hand"][0]["data"] == 2
    assert "NAGA" in p.strategy_view["me"]["board"][0]["races"]


def test_second_power_order_uses_explicit_entity_link():
    fixture = json.loads(gzip.decompress((Path(__file__).parent/"fixtures/client_match_20261030.json.gz").read_bytes()))
    snap = deepcopy(fixture["snapshots"]["alliance"])
    from tools.client_actions import bind_actions
    a = bind_actions(snap)
    old = a.zones["powers"][0]
    extra = deepcopy(old)
    extra["entity_id"] = 1  # Sorting by EID would put it first incorrectly.
    snap["entities"].append(extra)
    a.hero["tags"]["ADDITIONAL_HERO_POWER_ENTITY_1"] = 1
    a = bind_actions(snap)
    assert not a.issues
    assert [x["entity_id"] for x in a.zones["powers"]] == [old["entity_id"], 1]


@pytest.mark.parametrize("kind", ["quest", "timewarp"])
def test_nonstandard_choice_encoding_matches_native(kind):
    game = tg.Game(127, json.dumps({"hotfix_37_0_1": True}))
    for seat in range(8): game.step(seat, 30)
    ad = ClientFeatureAdapter()
    state = json.loads(game.state_json())
    def cid(name): return ad.vocabulary[name]-1
    if kind == "quest":
        q = next(k for k, v in ad.catalog.items() if v["kind"] == "quest")
        r = next(k for k, v in ad.catalog.items() if v["kind"] == "quest_reward")
        pending = {"Quest": {"options": [[cid(q), cid(r)]], "types": [[0, 0]]}}
    else:
        pending = {"Timewarp": {"options": [[{"Spell": {"card": cid("BG28_810"), "data": 0}}, 2]], "chronum": 7, "major": False}}
    state["players"][0]["pending"] = [pending]
    game = tg.Game.from_json(json.dumps(state))
    f, c, _, _ = encode_public_view(public(game), ad.vocabulary)
    nf, nc = game.observe(0)
    np.testing.assert_array_equal(f, np.frombuffer(nf, dtype="<f4"))
    np.testing.assert_array_equal(c, np.frombuffer(nc, dtype="<i4"))
