"""Read-only log integration: visibility, choices, partial writes and privacy."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("hslog")
from tools.client_state import newest_log, snapshot, watch_signature, write_snapshot


def line(message, method="DebugPrintPower"):
    return f"D 12:00:00.0000000 GameState.{method}() - {message}\n"


def start():
    return "".join(line(x) for x in [
        "CREATE_GAME", "GameEntity EntityID=1", "tag=CARDTYPE value=GAME",
        "tag=STATE value=RUNNING", "tag=TURN value=1",
        "Player EntityID=2 PlayerID=1 GameAccountId=[hi=0 lo=0]",
        "tag=CARDTYPE value=PLAYER", "tag=CONTROLLER value=1",
        "Player EntityID=3 PlayerID=2 GameAccountId=[hi=7 lo=987654321]",
        "tag=CARDTYPE value=PLAYER", "tag=CONTROLLER value=2"])


def card(entity=40, card_id="AT_121", controller=2, zone="HAND"):
    return "".join(line(x) for x in [
        f"FULL_ENTITY - Creating ID={entity} CardID={card_id}",
        "tag=CARDTYPE value=MINION", f"tag=CONTROLLER value={controller}",
        f"tag=ZONE value={zone}", "tag=ZONE_POSITION value=1", "tag=ATK value=4"])


def read(tmp_path, content, **kwargs):
    path = tmp_path / "Power.log"
    path.write_text(content)
    return snapshot(path, **kwargs)


def rows(result):
    assert result["status"] in ("snapshot", "incomplete_snapshot"), result
    return {r["entity_id"]: r for r in result["entities"]}


def test_records_revealed_state_and_redacts_accounts_and_hidden_identities(tmp_path):
    text = start() + card() + card(41, "", 1)
    text += line("TAG_CHANGE Entity=40 tag=ATK value=9")
    text += line("SHOW_ENTITY - Updating Entity=41 CardID=AT_121")
    text += line("HIDE_ENTITY - Entity=41 tag=ZONE value=HAND")
    result = read(tmp_path, text)
    entities = rows(result)
    assert result["perspective"]["player_id"] == 2
    assert entities[40]["tags"]["ATK"] == 9
    assert entities[40]["catalog_name_zh"] == "人气选手"
    assert entities[41]["card_id"] is None
    assert "987654321" not in json.dumps(result)
    assert "GameAccountId" not in json.dumps(result)
    assert not result["model_ready"]


def test_options_targets_and_choices_are_retained_and_invalidated(tmp_path):
    text = start() + card() + card(42, zone="PLAY")
    text += line("id=8", "DebugPrintOptions")
    text += line("option 0 type=END_TURN mainEntity= error=NONE errorParam=", "DebugPrintOptions")
    text += line("option 1 type=POWER mainEntity=40 error=NONE errorParam=", "DebugPrintOptions")
    text += line("target 0 entity=42 error=NONE errorParam=", "DebugPrintOptions")
    result = read(tmp_path, text)
    assert result["status"] == "snapshot", result
    assert result["options_after_last_state_change"]
    assert result["last_options"]["options"][1]["children"][0]["entity_id"] == 42
    text += line("selectedOption=1 selectedSubOption=-1 selectedTarget=42 selectedPosition=0", "SendOption")
    assert not read(tmp_path, text)["options_after_last_state_change"]
    text += line("id=9 Player=3 TaskList=1 ChoiceType=GENERAL CountMin=1 CountMax=1", "DebugPrintEntityChoices")
    text += line("Source=40", "DebugPrintEntityChoices")
    text += line("Entities[0]=[entityName=Test id=42 zone=PLAY zonePos=1 cardId=AT_121 player=2]", "DebugPrintEntityChoices")
    result = read(tmp_path, text)
    assert result["last_choices"]["entity_ids"] == [42], result
    assert not result["last_choices"]["selection_sent_or_resolved"]
    text += line("id=9 ChoiceType=GENERAL", "SendChoices")
    assert read(tmp_path, text)["last_choices"]["selection_sent_or_resolved"]


def test_incomplete_tail_does_not_apply_half_written_tag(tmp_path):
    text = start() + card() + line("TAG_CHANGE Entity=40 tag=ATK value=999").rstrip("\n")
    result = read(tmp_path, text)
    assert result["status"] == "incomplete_snapshot"
    assert rows(result)[40]["tags"]["ATK"] == 4
    assert result["source"]["partial_trailing_line"]


def test_open_block_is_reported_as_incomplete(tmp_path):
    text = start() + card()
    text += line("BLOCK_START BlockType=PLAY Entity=40 EffectCardId= EffectIndex=0 Target=0 SubOption=-1")
    result = read(tmp_path, text)
    assert result["status"] == "incomplete_snapshot", result
    assert result["open_blocks"] == 1
    assert read(tmp_path, text + line("BLOCK_END"))["status"] == "snapshot"


def test_latest_game_and_rotated_log_do_not_reuse_prior_state(tmp_path):
    result = read(tmp_path, start() + card(40) + start() + card(55))
    assert 55 in rows(result) and 40 not in rows(result)
    newer = tmp_path / "new-session" / "Power.log"
    newer.parent.mkdir()
    newer.write_text("")
    os.utime(newer, ns=(2_000_000_000_000_000_000, 2_000_000_000_000_000_000))
    assert newest_log(tmp_path) == newer
    assert snapshot(newer)["status"] == "empty_log"
    assert snapshot(None)["status"] == "no_log"


def test_unknown_card_and_build_are_reported_without_inventing_support(tmp_path):
    result = read(tmp_path, start() + card(card_id="NEW_UNKNOWN_CARD"), installed_build="37.0.255790")
    assert rows(result)[40]["catalog_name"] is None
    assert result["unknown_card_ids"] == ["NEW_UNKNOWN_CARD"]
    assert not result["model_ready"] and not result["versions"]["log_build_verified"]
    assert "installed_client_differs_from_simulator" in result["model_blockers"]


def test_corrupt_log_error_does_not_echo_private_text(tmp_path):
    result = read(tmp_path, start() + "PrivatePlayer#1234 invalid line\n")
    assert result["status"] == "read_or_parse_error"
    assert "PrivatePlayer" not in json.dumps(result)
    assert not result["model_ready"]
    assert read(tmp_path, start() + card(), player_id=99)["status"] == "read_or_parse_error"


def test_snapshot_file_is_local_private_json(tmp_path):
    result = read(tmp_path, start() + card(), player_id=2)
    target = tmp_path / "out" / "state.json"
    write_snapshot(target, result)
    assert json.loads(target.read_text()) == result
    assert target.stat().st_mode & 0o777 == 0o600
    assert not target.with_suffix(".json.tmp").exists()


def test_idle_log_expires_and_changes_watch_signature_without_new_bytes(tmp_path):
    path = tmp_path / "Power.log"
    path.write_text(start() + card())
    observed = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    fresh = snapshot(path, now=observed)
    old = snapshot(path, now=observed + timedelta(seconds=31))
    assert fresh["freshness"]["status"] == "recent"
    assert old["freshness"]["status"] == "stale" and "log_stale" in old["model_blockers"]
    assert fresh["snapshot_id"] == old["snapshot_id"]
    assert watch_signature(path, now=observed) != watch_signature(path, now=observed + timedelta(seconds=31))
    assert not old["model_ready"] and not old["live_ready"]


def test_newest_named_session_without_log_never_revives_previous_game(tmp_path):
    old = tmp_path / "Hearthstone_2026_10_29_21_00_00/Power.log"
    old.parent.mkdir()
    old.write_text(start() + card())
    current = tmp_path / "Hearthstone_2026_10_30_08_00_00/Power.log"
    current.parent.mkdir()
    assert newest_log(tmp_path) == current
    result = snapshot(newest_log(tmp_path))
    assert result["status"] == "missing_log" and "entities" not in result
    current.write_text("")
    os.utime(old, (2_000_000_000, 2_000_000_000))
    assert newest_log(tmp_path) == current


def test_log_replacement_during_read_is_incomplete_even_if_size_and_mtime_match(tmp_path, monkeypatch):
    path = tmp_path / "Power.log"
    text = start() + card()
    path.write_text(text)
    before = path.stat()
    original = os.fstat
    calls = 0
    def replaced(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.unlink()
            path.write_text(text)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return original(fd)
    monkeypatch.setattr(os, "fstat", replaced)
    result = snapshot(path)
    assert result["status"] == "incomplete_snapshot"
    assert result["source"]["changed_while_reading"]
    assert "incomplete_log" in result["model_blockers"]


def test_unresolved_choices_become_stale_after_state_change(tmp_path):
    text = start() + card()
    text += line("id=9 Player=3 TaskList=1 ChoiceType=GENERAL CountMin=1 CountMax=1", "DebugPrintEntityChoices")
    text += line("Source=40", "DebugPrintEntityChoices")
    text += line("Entities[0]=[entityName=Test id=40 zone=HAND zonePos=1 cardId=AT_121 player=2]", "DebugPrintEntityChoices")
    result = read(tmp_path, text)
    assert result["choices_after_last_state_change"]
    result = read(tmp_path, text + line("TAG_CHANGE Entity=40 tag=ATK value=5"))
    assert not result["choices_after_last_state_change"]
    assert "stale_choices" in result["model_blockers"]


def test_unknown_cards_and_missing_actions_have_explicit_blockers(tmp_path):
    result = read(tmp_path, start() + card(card_id="UNKNOWN"))
    assert {"unknown_cards", "missing_current_options", "perspective_unconfirmed"} <= set(result["model_blockers"])
