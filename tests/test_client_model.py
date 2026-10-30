"""Model boundary: missing inputs, entity actions, decoder history and stale work."""
from copy import deepcopy
import json
import subprocess
import sys
import time
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
import taverngym as tg
from tools.client_actions import bind_actions, frame_identity, render_bound_action, match_submission
from tools.client_features import ClientFeatureAdapter
from tools.client_model import ModelUnavailable, PolicyRuntime, native_observation
from tools.client_model_bridge import ModelBridge
from training.entity import EntityPolicy


def entity(eid, cid, kind, owner, position=None, zone="PLAY", **tags):
    return {"entity_id": eid, "card_id": cid, "type": kind, "controller": owner, "zone": zone,
            "position": position, "catalog_name_zh": cid, "tags": tags}


def option(i, eid, *, error=None, kind="POWER", targets=()):
    return {"id": i, "entity_id": eid, "type": kind, "kind": "option", "error": error,
            "children": [{"id": j, "entity_id": target, "kind": "target", "error": err, "children": []}
                         for j, (target, err) in enumerate(targets)]}


def frame():
    return {"snapshot_id": "first", "status": "snapshot", "game_status": "running",
        "freshness": {"status": "recent"}, "perspective": {"player_id": 3},
        "source": {"path": "/game/Power.log", "device": 1, "inode": 2, "game_start_line": 1},
        "entities": [entity(1, None, "GAME", None, ALLOW_MOVE_MINION=1),
            entity(2, None, "PLAYER", 3, PLAYER_ID=3, HERO_ENTITY=20, PLAYSTATE=1,
                   PLAYER_TECH_LEVEL=2, RESOURCES=6, RESOURCES_USED=2, TURN=3),
            entity(3, None, "PLAYER", 14, PLAYER_ID=14, HERO_ENTITY=21),
            entity(20, "TB_BaconShop_HERO_01", "HERO", 3, HEALTH=30, DAMAGE=2, ARMOR=10),
            entity(21, "TB_BaconShopBob", "HERO", 14),
            entity(30, "AT_121", "MINION", 14, 1, ATK=2, HEALTH=3, TECH_LEVEL=1),
            entity(31, "BGS_004", "MINION", 14, 2, ATK=1, HEALTH=1, TECH_LEVEL=1),
            entity(40, "AT_121", "MINION", 3, 1, ATK=2, HEALTH=3, TECH_LEVEL=1),
            entity(41, "BGS_004", "MINION", 3, 2, ATK=1, HEALTH=1, TECH_LEVEL=1),
            entity(50, "BG28_504", "BATTLEGROUND_SPELL", 3, 1, "HAND"),
            entity(60, "TB_BaconShop_DragBuy", "MOVE_MINION_HOVER_TARGET", 3, COST=3),
            entity(61, "TB_BaconShopLockAll_Button", "GAME_MODE_BUTTON", 3)],
        "options_after_last_state_change": True, "last_options": {"id": 9, "options": [
            option(0, None, error="INVALID", kind="END_TURN"),
            option(1, 60, targets=[(30, None), (31, "REQ_ENOUGH_MANA")]),
            option(2, 30), option(3, 31), option(4, 40), option(5, 41),
            option(6, 50, targets=[(40, None), (41, None)]), option(7, 61)]}}


def test_transactions_moves_freeze_and_shop_dragging():
    snap = frame()
    a = bind_actions(snap)
    assert set(a.bindings) == {0, 2, 8, 46, 197, 203}
    assert a.bindings[8]["source_entity_id"] == 60
    assert a.bindings[8]["target_entity_id"] == 30
    assert 9 not in a.bindings
    assert a.bindings[197]["source_entity_id"] == 40
    assert a.bindings[203]["source_entity_id"] == 41
    assert a.bindings[0]["kind"] == "wait_for_combat"
    assert render_bound_action(a, 2)["kind"] == "freeze"
    for e in snap["entities"]:
        if e["entity_id"] in (30, 31):
            e["tags"]["FROZEN"] = 1
    assert bind_actions(snap).bindings[2]["kind"] == "unfreeze"


def test_targeted_spell_never_gets_arbitrary_first_target():
    a = bind_actions(frame())
    assert a.bindings[46]["targets"] == [40, 41]
    with pytest.raises(ValueError, match="second-stage"):
        render_bound_action(a, 46)
    with pytest.raises(ValueError, match="bound"):
        render_bound_action(a, 9)


def test_choice_packet_order_replaces_options_mask():
    snap = frame()
    snap["last_choices"] = {"id": 18, "player_entity_id": 2, "source_entity_id": 50, "type": "GENERAL",
        "min": 1, "max": 1, "entity_ids": [41, 40], "selection_sent_or_resolved": False}
    snap["choices_after_last_state_change"] = True
    a = bind_actions(snap)
    assert set(a.bindings) == {30, 31}
    assert a.bindings[30]["source_entity_id"] == 41
    assert a.bindings[31]["source_entity_id"] == 40
    assert a.choice_kind == "discover"
    snap["last_choices"]["max"] = 2
    assert not bind_actions(snap).bindings


def test_phase_and_session_invalidate_identity():
    snap = frame()
    old = frame_identity(snap)
    snap["source"]["game_start_line"] = 20
    assert frame_identity(snap) != old
    snap["entities"][0]["tags"]["BACON_IN_COMBAT_PHASE"] = 1
    assert bind_actions(snap).bindings == {}
    assert bind_actions(snap).zones["shop"] == []


def test_duplicate_modes_are_not_overwritten_or_treated_as_executed():
    snap = frame()
    snap["last_options"]["options"].append(option(8, 50))
    a = bind_actions(snap)
    assert a.bindings[46]["option_id"] == 6
    result = match_submission(a, {"kind": "cast_spell", "options_id": 9, "option": 8,
                                  "main": {"entity_id": 50}, "target": {"entity_id": 0}})
    assert result["status"] == "requires_mode_mapping" and not result["execution_verified"]


def test_submission_requires_exact_option_target_position_and_choice_identity():
    a = bind_actions(frame())
    event = {"kind": "buy_minion", "options_id": 9, "option": 1,
             "main": {"entity_id": 60}, "target": {"entity_id": 30}}
    assert match_submission(a, event)["action"] == 8
    event["target"]["entity_id"] = 31
    assert match_submission(a, event)["status"] == "unmapped"
    event = {"kind": "cast_spell", "options_id": 9, "option": 6,
             "main": {"entity_id": 50}, "target": {"entity_id": 41}}
    assert match_submission(a, event)["status"] == "requires_target_decision"
    event["options_id"] = 10
    assert match_submission(a, event)["status"] == "unmapped"


def test_unknown_features_and_no_opponent_hands():
    snap = frame()
    snap["entities"].append(entity(90, "BG29_888", "MINION", 14, 1, "HAND", ATK=900, HEALTH=900))
    adapter = ClientFeatureAdapter()
    packet = adapter.convert(snap)
    assert packet.features.shape == (3227,) and packet.features.dtype == np.float32
    assert packet.cards.shape == (124,) and packet.cards.dtype == np.int32
    assert packet.action_mask.shape == (245,) and packet.action_mask.dtype == np.bool_
    assert packet.features[7] == np.float32(.4)
    assert packet.features[5] == np.float32(28 / 40)
    assert packet.features[285 + 2] == 0
    assert np.isnan(packet.features[42:53]).all()
    assert not packet.complete and packet.growth is None and packet.attachments is None
    assert adapter.vocabulary["BG29_888"] not in packet.cards
    assert np.isnan(packet.features[1708:]).all()
    json.dumps(packet.report(), allow_nan=False)


@pytest.fixture
def runtime(tmp_path):
    model = EntityPolicy(width=16, layers=1, heads=2, text_dim=8, attachment_version=1, growth_version=1)
    checkpoint = tmp_path / "contract-test.pt"
    torch.save({"engine_hash": tg.source_hash(), "architecture": model.architecture, "model": model.state_dict(),
                "args": {"action_cap": 128, "hotfix_37_0_1": True}}, checkpoint)
    return PolicyRuntime(checkpoint, threads=1)


def test_checkpoint_identity(runtime, tmp_path):
    with pytest.raises(ModelUnavailable, match="checksum"):
        PolicyRuntime(runtime.path, expected_sha256="0" * 64)
    data = torch.load(runtime.path, weights_only=True)
    data["engine_hash"] = "different-engine"
    p = tmp_path / "other.pt"
    torch.save(data, p)
    with pytest.raises(ValueError, match="engine"):
        PolicyRuntime(p)


def test_required_auxiliary_inputs_and_mask(runtime):
    game = tg.Game(123)
    obs, view = native_observation(game, 0, runtime)
    result = runtime.infer_arrays(obs, strategy_view=view)
    assert obs["action_mask"][0, result.action] and result.probability > 0 and result.latency_ms > 0
    assert runtime.guard.save(0)["state"] is None
    assert runtime.infer_arrays(obs, strategy_view=view).action == result.action
    for key in ("growth", "attachments"):
        missing = dict(obs)
        del missing[key]
        with pytest.raises(ValueError, match=key.rstrip("s")):
            runtime.infer_arrays(missing)
    bad = {**obs, "action_mask": np.zeros((1, 245), np.bool_)}
    with pytest.raises(ModelUnavailable, match="empty"):
        runtime.infer_arrays(bad)
    bad = {**obs, "features": obs["features"].copy()}
    bad["features"][0, 42] = np.nan
    with pytest.raises(ModelUnavailable, match="Incomplete"):
        runtime.infer_arrays(bad)
    with pytest.raises(ModelUnavailable, match="incomplete"):
        runtime.infer_client(ClientFeatureAdapter().convert(frame()))


def test_position_preview_restores_history_on_success_and_failure(runtime, monkeypatch):
    from training.position import PublicPositionPolicy, FAMILY
    encoder = EntityPolicy(width=16, layers=1, heads=2, text_dim=8)
    size = 6 * 16 + 51 + 7 * 31
    ridge = {"family": FAMILY, "weight": torch.zeros(size), "feature_mean": torch.zeros(size),
             "feature_scale": torch.ones(size), "intercept": 0.}
    model = PublicPositionPolicy(runtime.policy, encoder, ridge)
    runtime.policy, runtime.position = model, True
    obs, view = native_observation(tg.Game(123), 0, runtime)
    before = (deepcopy(model.save_context()), deepcopy(model.counts), list(model.move_events))
    assert runtime.infer_arrays(obs).action == runtime.infer_arrays(obs).action
    assert (model.save_context(), model.counts, model.move_events) == before
    def fail(*args, **kwargs):
        raise RuntimeError("failed actor")
    monkeypatch.setattr(model.actor, "forward", fail)
    with pytest.raises(RuntimeError, match="failed actor"):
        runtime.infer_arrays(obs)
    assert model.context is None
    assert (model.save_context(), model.counts, model.move_events) == before
    assert runtime.guard.save(0)["state"] is None


def wait_for(check, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = check()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("Timed out waiting for model bridge")


def test_latest_frame_worker_drops_late_results_without_blocking_reader(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text("""import json,sys,time
from tools.client_actions import frame_identity
print(json.dumps({'status':'loaded','policy':{'update':15}}),flush=True)
for line in sys.stdin:
 s=json.loads(line)['snapshot'];time.sleep(.1)
 print(json.dumps({'status':'input_incomplete','frame_id':frame_identity(s),'input':{'source':s['snapshot_id']}}),flush=True)
""")
    def launch(command, **kwargs):
        return subprocess.Popen([sys.executable, "-u", "-c", script.read_text()], **kwargs)
    bridge = ModelBridge({"checkpoint": "unused"}, launch=launch)
    try:
        wait_for(lambda: bridge.policy)
        a = frame()
        before = time.perf_counter()
        assert bridge.sample(a)["status"] == "converting"
        assert time.perf_counter() - before < .08
        wait_for(lambda: bridge.pending is None)
        b = deepcopy(a); b["snapshot_id"] = "second"
        c = deepcopy(a); c["snapshot_id"] = "third"
        bridge.sample(b); bridge.sample(c)
        result = wait_for(lambda: (r if (r := bridge.sample(c)).get("status") == "input_incomplete" else None))
        assert result["input"]["source"] == "third"
        assert bridge.sample(c, enabled=False)["status"] == "waiting_for_game"
        assert bridge.current_key is None
    finally:
        bridge.close()
    assert bridge.process.poll() is not None
