"""Inspect a macOS Hearthstone installation and log metadata without changing it."""

import argparse
import configparser
from datetime import datetime, timezone
import json
from pathlib import Path
import plistlib
import subprocess

if __package__:
    from .client_common import load_catalog, newest_log, write_private_json
else:
    from client_common import load_catalog, newest_log, write_private_json


def app_info(app):
    result = {"path": str(app), "status": "unreadable", "build": None}
    try:
        info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
        if not isinstance(info, dict):
            return result
        result.update(status="found", build=info.get("CFBundleVersion"),
                      bundle_id=info.get("CFBundleIdentifier"))
    except (OSError, ValueError, plistlib.InvalidFileException):
        pass
    return result


def logging_info(config):
    result = {"path": str(config), "status": "missing", "power_file_enabled": False}
    try:
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(config.read_text(encoding="utf-8-sig"))
        enabled = parser.getboolean("Power", "FilePrinting", fallback=False)
        verbose = parser.getboolean("Power", "Verbose", fallback=False)
        level = parser.getint("Power", "LogLevel", fallback=None)
        result.update(status="read", power_file_enabled=enabled and level == 1,
                      file_printing=enabled, verbose=verbose, log_level=level)
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, ValueError, configparser.Error):
        result["status"] = "unreadable"
    return result


def discover_apps():
    roots = [Path("/Applications"), Path.home() / "Applications"]
    volumes = Path("/Volumes")
    if volumes.is_dir():
        roots += [p for p in volumes.iterdir() if p.is_dir()]
    candidates = [p for root in roots for p in
                  (root / "Hearthstone/Hearthstone.app", root / "Hearthstone.app") if p.is_dir()]
    return sorted(set(p.resolve() for p in candidates))


def client_running():
    try:
        result = subprocess.run(["ps", "-axo", "comm="], check=True, capture_output=True,
                                text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return any(p.strip().endswith("/Hearthstone.app/Contents/MacOS/Hearthstone")
               for p in result.stdout.splitlines())


def audit(*, apps, logs, config, running=None):
    meta, _ = load_catalog()
    result = {"schema": "autotavern.client-audit.v1", "scope": "read_only_local_audit",
              "captured_at": datetime.now(timezone.utc).isoformat(),
              "model_ready": False, "live_ready": False,
              "installations": [app_info(app) for app in apps],
              "client_running": running, "logging": logging_info(config),
              "simulator_client": meta["client_version"], "log_build_verified": False,
              "logs": {"directory": str(logs) if logs else None, "latest": None, "files": []},
              "issues": []}
    if not apps:
        result["issues"].append("client_not_found")
    if len(apps) > 1:
        result["issues"].append("multiple_installations_select_client_app")
    for app in result["installations"]:
        if app["status"] != "found" or not app["build"]:
            result["issues"].append("client_build_unreadable")
        elif app["build"] != meta["client_version"]:
            result["issues"].append("installed_client_differs_from_simulator")
    if not result["logging"]["power_file_enabled"]:
        result["issues"].append("power_logging_not_confirmed")
    elif not result["logging"].get("verbose"):
        result["issues"].append("power_verbose_not_enabled")
    if running is not True:
        result["issues"].append("client_not_running" if running is False else "client_process_unknown")
    try:
        latest = newest_log(logs) if logs else None
        result["logs"]["latest"] = str(latest) if latest else None
        paths = sorted(logs.glob("*/Power.log")) if logs else []
        if logs and (logs / "Power.log").exists():
            paths.append(logs / "Power.log")
        for path in paths:
            stat = path.stat()
            result["logs"]["files"].append({"path": str(path), "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
        if not latest or not latest.is_file():
            result["issues"].append("latest_power_log_missing")
        elif latest.stat().st_size == 0:
            result["issues"].append("latest_power_log_empty")
    except OSError:
        result["issues"].append("log_metadata_unreadable")
    result["issues"] += ["log_build_and_server_rules_unverified", "live_adapter_unvalidated",
                         "model_readiness_not_established"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-app", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path.home() / "Library/Preferences/Blizzard/Hearthstone/log.config")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    apps = [args.client_app] if args.client_app else discover_apps()
    logs = args.logs or (apps[0].parent / "Logs" if len(apps) == 1 else None)
    result = audit(apps=apps, logs=logs, config=args.config, running=client_running())
    if args.out:
        protected = [args.config] + [app / "Contents/Info.plist" for app in apps]
        protected += [Path(row["path"]) for row in result["logs"]["files"]]
        if args.out.resolve() in {path.resolve() for path in protected}:
            parser.error("--out must not overwrite client configuration or logs")
        write_private_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
