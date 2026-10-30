"""Incremental equivalence across writes, mutable packets and session boundaries."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import os

import pytest

from tools.client_state import snapshot
from tools.client_stream import IncrementalReader


def test_enum_name_cache_does_not_confuse_equal_numbers_from_different_enums():
    from hearthstone.enums import ChoiceType, OptionType
    from tools.client_state import enum_name
    enum_name.cache_clear()
    assert enum_name(OptionType.END_TURN) == "END_TURN"
    assert enum_name(ChoiceType.GENERAL) == "GENERAL"


def line(text, method="DebugPrintPower"):
    return f"D 12:00:00.0000000 GameState.{method}() - {text}\n"


def game(entity=40):
    return "".join(line(t) for t in ["CREATE_GAME", "GameEntity EntityID=1",
        "tag=CARDTYPE value=GAME", "tag=STATE value=RUNNING",
        "Player EntityID=2 PlayerID=1 GameAccountId=[hi=0 lo=0]",
        "tag=CARDTYPE value=PLAYER", "tag=CONTROLLER value=1",
        "Player EntityID=3 PlayerID=2 GameAccountId=[hi=7 lo=987654321]",
        "tag=CARDTYPE value=PLAYER", "tag=CONTROLLER value=2",
        f"FULL_ENTITY - Creating ID={entity} CardID=AT_121", "tag=CARDTYPE value=MINION",
        "tag=CONTROLLER value=2", "tag=ZONE value=HAND", "tag=ZONE_POSITION value=1",
        "tag=ATK value=4", "tag=HEALTH value=4"])


COMPARE = ("status", "entities", "players", "perspective", "last_options", "last_choices",
           "options_after_last_state_change", "choices_after_last_state_change", "open_blocks",
           "recent_actions", "action_counts", "game_status", "parser_diagnostics")


def assert_batch_matches(reader, path):
    actual, expected = reader.read(path), snapshot(path)
    for key in COMPARE:
        assert actual.get(key) == expected.get(key), (key, actual.get("error"), expected.get("error"))
    return actual


@pytest.mark.parametrize("size", [1, 37, 257, 4096])
def test_split_writes_preserve_tags_options_choices_and_inputs(tmp_path, size):
    content = game()
    content += line("id=8", "DebugPrintOptions")
    content += line("option 0 type=POWER mainEntity=40 error=NONE errorParam=", "DebugPrintOptions")
    content += line("target 0 entity=40 error=NONE errorParam=", "DebugPrintOptions")
    content += line("selectedOption=0 selectedSubOption=-1 selectedTarget=0 selectedPosition=1", "SendOption")
    content += line("TAG_CHANGE Entity=40 tag=ZONE value=PLAY")
    content += line("id=9 Player=3 TaskList=1 ChoiceType=GENERAL CountMin=1 CountMax=1", "DebugPrintEntityChoices")
    content += line("Source=40", "DebugPrintEntityChoices")
    content += line("Entities[0]=[entityName=PrivateName id=40 zone=PLAY zonePos=1 cardId=AT_121 player=2]", "DebugPrintEntityChoices")
    content += line("id=9 ChoiceType=GENERAL", "SendChoices")
    content += line("m_chosenEntities[0]=[entityName=PrivateName id=40 zone=PLAY zonePos=1 cardId=AT_121 player=2]", "SendChoices")
    content += line("TAG_CHANGE Entity=40 tag=ATK value=9")
    content += line("HIDE_ENTITY - Entity=40 tag=ZONE value=HAND")
    path = tmp_path / "Power.log"
    path.touch()
    reader = IncrementalReader()
    raw = content.encode()
    for offset in range(0, len(raw), size):
        with path.open("ab") as stream: stream.write(raw[offset:offset + size])
        state = assert_batch_matches(reader, path)
    assert state["action_counts"] == {"play_minion": 1, "choice": 1}
    assert state["entities"][-1]["card_id"] is None
    assert "PrivateName" not in str(state) and "987654321" not in str(state)


def test_idle_and_incremental_reads_do_not_reconsume_game(tmp_path):
    path = tmp_path / "Power.log"
    path.write_text(game())
    reader = IncrementalReader()
    first = reader.read(path)
    saved = deepcopy(first)
    assert reader.read(path)["performance"]["read_bytes"] == 0
    addition = line("TAG_CHANGE Entity=40 tag=ATK value=9")
    with path.open("a") as stream: stream.write(addition)
    updated = reader.read(path)
    assert updated["performance"]["read_bytes"] == len(addition.encode())
    assert updated["entities"][-1]["tags"]["ATK"] == 9
    assert first == saved
    later = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc) + timedelta(seconds=31)
    assert reader.read(path, now=later)["freshness"]["status"] == "stale"


def test_submission_needs_server_transition_before_execution_confirmation(tmp_path):
    path = tmp_path / "Power.log"
    text = game() + line("id=8", "DebugPrintOptions")
    text += line("option 0 type=POWER mainEntity=40 error=NONE errorParam=", "DebugPrintOptions")
    text += line("selectedOption=0 selectedSubOption=-1 selectedTarget=0 selectedPosition=1", "SendOption")
    path.write_text(text)
    reader = IncrementalReader()
    state = reader.read(path)
    assert not state["recent_actions"][-1].get("execution_verified")
    with path.open("a") as stream:
        stream.write(line("TAG_CHANGE Entity=40 tag=ZONE value=PLAY"))
        stream.write(line("id=9", "DebugPrintOptions"))
        stream.write(line("option 0 type=END_TURN mainEntity=0 error=NONE errorParam=", "DebugPrintOptions"))
    state = reader.read(path)
    assert state["recent_actions"][-1]["execution_verified"]
    assert state["public_history"]["stats"]["2"]["cards_played"] == 1
    # Re-reading a growing menu must not count the same action twice.
    with path.open("a") as stream:
        stream.write(line("option 1 type=POWER mainEntity=40 error=NONE errorParam=", "DebugPrintOptions"))
    assert reader.read(path)["public_history"]["stats"]["2"]["cards_played"] == 1


@pytest.mark.parametrize("change", ["new_game", "replace", "truncate", "rewrite", "missing", "new_session"])
def test_lifecycle_never_carries_previous_entities_or_actions(tmp_path, change):
    path = tmp_path / "Power.log"
    path.write_text(game())
    reader = IncrementalReader()
    reader.read(path)
    new = game(55)
    if change == "new_game":
        with path.open("a") as stream: stream.write(new)
    elif change == "replace":
        other = tmp_path / "other"; other.write_text(new); other.replace(path)
    elif change == "truncate":
        path.write_text("")
        assert reader.read(path)["status"] == "empty_log"
        path.write_text(new)
    elif change == "missing":
        path.unlink()
        assert reader.read(path)["status"] == "missing_log"
        path.write_text(new)
    elif change == "new_session":
        path = tmp_path / "new-session.log"; path.write_text(new)
    else:
        path.write_text(new)
    state = assert_batch_matches(reader, path)
    assert {e["entity_id"] for e in state["entities"]} == {1, 2, 3, 55}
    assert state["action_counts"] == {}


def test_compatibility_warning_is_counted_once_and_open_blocks_are_explicit(tmp_path):
    path = tmp_path / "Power.log"
    path.write_text(game() + line("BLOCK_START BlockType=PLAY Entity=40 EffectCardId= EffectIndex=0 Target=0 SubOption=-1"))
    reader = IncrementalReader()
    assert assert_batch_matches(reader, path)["open_blocks"] == 1
    with path.open("a") as stream:
        stream.write(line("id=1", "DebugPrintOptions"))
    state = assert_batch_matches(reader, path)
    assert state["parser_diagnostics"]["warning_counts"] == {"battlegrounds_missing_block_end": 1}
    assert reader.read(path)["parser_diagnostics"] == state["parser_diagnostics"]


def test_corrupt_prefix_stays_blocked_until_a_new_game(tmp_path):
    path = tmp_path / "Power.log"; path.write_text(game() + "PrivateAccount broken line\n")
    reader = IncrementalReader()
    assert reader.read(path)["status"] == "read_or_parse_error"
    with path.open("a") as stream: stream.write(game(55))
    assert assert_batch_matches(reader, path)["status"] == "snapshot"


def test_internal_entities_are_separate_but_unknown_gameplay_cards_still_block(tmp_path):
    path = tmp_path / "Power.log"
    text = game().replace("CardID=AT_121", "CardID=UNSEEN_MINION")
    for eid, cid, kind in [(41, "Button", "GAME_MODE_BUTTON"), (42, "Effect", "ENCHANTMENT"), (43, "Skin", "HERO")]:
        text += line(f"FULL_ENTITY - Creating ID={eid} CardID={cid}") + line(f"tag=CARDTYPE value={kind}")
        if eid == 43: text += line("tag=BACON_SKIN value=1")
    path.write_text(text)
    state = assert_batch_matches(IncrementalReader(), path)
    assert state["unknown_card_ids"] == ["UNSEEN_MINION"]
    assert state["unmapped_internal_ids"] == ["Button", "Effect"]
    assert state["unmapped_skin_ids"] == ["Skin"]
    assert "unknown_cards" in state["model_blockers"]
    assert "internal_entity_mapping_unvalidated" in state["model_blockers"]


def test_source_replacement_during_read_is_not_published_as_stable(tmp_path, monkeypatch):
    path = tmp_path / "Power.log"; path.write_text(game())
    original = os.fstat
    calls = 0
    def replace(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement = tmp_path / "other"; replacement.write_text(game(55)); replacement.replace(path)
        return original(fd)
    monkeypatch.setattr(os, "fstat", replace)
    reader = IncrementalReader()
    state = reader.read(path)
    assert state["status"] == "incomplete_snapshot"
    assert state["source"]["changed_while_reading"]
    assert assert_batch_matches(reader, path)["entities"][-1]["entity_id"] == 55


def test_background_leaderboard_changes_preserve_own_options_but_resources_invalidate(tmp_path):
    path = tmp_path / "Power.log"
    content = game()
    content += "".join(line(s) for s in ["FULL_ENTITY - Creating ID=50 CardID=TB_BaconShop_HERO_01",
        "tag=CARDTYPE value=HERO", "tag=CONTROLLER value=8", "tag=ZONE value=SETASIDE", "tag=ARMOR value=10"])
    content += line("id=8", "DebugPrintOptions") + line("option 0 type=POWER mainEntity=40 error=NONE errorParam=", "DebugPrintOptions")
    content += line("TAG_CHANGE Entity=50 tag=ARMOR value=9")
    path.write_text(content)
    reader = IncrementalReader()
    assert assert_batch_matches(reader, path)["options_after_last_state_change"]
    with path.open("a") as stream: stream.write(line("TAG_CHANGE Entity=2 tag=RESOURCES value=4"))
    assert not assert_batch_matches(reader, path)["options_after_last_state_change"]


def test_mutable_menus_wait_for_packet_boundary_or_quiet_eof(tmp_path):
    clock = [100.]
    path = tmp_path / "Power.log"
    path.write_text(game() + line("id=8", "DebugPrintOptions") +
        line("option 0 type=POWER mainEntity=40 error=NONE errorParam=", "DebugPrintOptions"))
    reader = IncrementalReader(clock=lambda: clock[0])
    assert reader.read(path)["options_packet_complete"] is False
    clock[0] += .01
    assert reader.read(path)["options_packet_complete"] is False
    with path.open("a") as stream: stream.write(line("target 0 entity=40 error=NONE errorParam=", "DebugPrintOptions"))
    assert reader.read(path)["options_packet_complete"] is False
    clock[0] += .03
    settled = reader.read(path)
    assert settled["options_packet_complete"] and settled["performance"]["read_bytes"] == 0
    with path.open("a") as stream:
        stream.write(line("id=9 Player=3 TaskList=1 ChoiceType=GENERAL CountMin=1 CountMax=1", "DebugPrintEntityChoices"))
        stream.write(line("Source=40", "DebugPrintEntityChoices"))
        stream.write(line("Entities[0]=[entityName=Minion id=40 zone=HAND zonePos=1 cardId=AT_121 player=2]", "DebugPrintEntityChoices"))
    assert reader.read(path)["choice_packet_complete"] is False
    with path.open("a") as stream: stream.write(line("TAG_CHANGE Entity=40 tag=ATK value=5"))
    assert reader.read(path)["choice_packet_complete"] is True
