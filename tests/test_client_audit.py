"""Local installation checks only; never launch the game or read private logs."""

import json
import plistlib

from tools.client_audit import audit, logging_info
from tools.client_common import write_private_json


def test_audit_reports_build_settings_and_empty_logs_without_writing_them(tmp_path):
    app = tmp_path / "Hearthstone.app"
    (app / "Contents").mkdir(parents=True)
    (app / "Contents/Info.plist").write_bytes(plistlib.dumps({"CFBundleVersion": "37.0.255790"}))
    config = tmp_path / "log.config"
    config.write_text("[Power]\nLogLevel=1\nFilePrinting=true\nVerbose=true\n[Other]\nSecret=PRIVATE\n")
    log = tmp_path / "Logs/Hearthstone_2026_10_30_09_00_00/Power.log"
    log.parent.mkdir(parents=True)
    log.touch()
    before = config.read_bytes(), log.stat().st_mtime_ns
    result = audit(apps=[app], logs=log.parent.parent, config=config, running=False)
    assert result["logging"]["power_file_enabled"]
    assert {"latest_power_log_empty", "client_not_running", "installed_client_differs_from_simulator"} <= set(result["issues"])
    assert "PRIVATE" not in json.dumps(result) and not result["model_ready"]
    assert before == (config.read_bytes(), log.stat().st_mtime_ns)


def test_invalid_or_absent_configuration_is_actionable_and_private(tmp_path):
    config = tmp_path / "log.config"
    assert logging_info(config)["status"] == "missing"
    config.write_text("[Power]\nFilePrinting=PRIVATE_INVALID\n")
    result = logging_info(config)
    assert result["status"] == "unreadable" and "PRIVATE_INVALID" not in json.dumps(result)
    result = audit(apps=[], logs=None, config=config)
    assert {"client_not_found", "power_logging_not_confirmed", "latest_power_log_missing"} <= set(result["issues"])


def test_atomic_writer_does_not_follow_predictable_temporary_symlink(tmp_path):
    victim = tmp_path / "preserved.txt"
    victim.write_text("original")
    path = tmp_path / "state.json"
    path.with_suffix(".json.tmp").symlink_to(victim)
    write_private_json(path, {"private": True})
    assert victim.read_text() == "original"
    assert path.stat().st_mode & 0o777 == 0o600
