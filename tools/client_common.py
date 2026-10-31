"""Local recording utilities; no simulator, policy or client-control dependencies."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile


CATALOG = Path(__file__).resolve().parents[1] / "crates/taverngym/data/cards.json"
DEFAULT_MAX_AGE = 30.0
SESSION = re.compile(r"Hearthstone_\d{4}(?:_\d{2}){5}$")


def newest_log(directory):
    """Never fall back to an old game when the newest client session has no log."""
    sessions = [p for p in directory.iterdir() if p.is_dir() and SESSION.fullmatch(p.name)] \
        if directory.is_dir() else []
    if sessions:
        return max(sessions, key=lambda p: p.name) / "Power.log"
    candidates = [p for p in directory.glob("*/Power.log") if p.is_file()]
    if (directory / "Power.log").is_file():
        candidates.append(directory / "Power.log")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)), default=None)


def content_id(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def log_game_key(source):
    """Identify one game within a particular client log, including file rotation."""
    if not source.get("game_start_line"):
        return None
    return content_id([source.get(k) for k in ("path", "device", "inode", "game_start_line")])


def freshness(observed_at, *, now=None, max_age=DEFAULT_MAX_AGE):
    """A recording age check, never proof that the game still shows this state."""
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("max_age must be finite and positive")
    now = now or datetime.now(timezone.utc)
    result = {"status": "unknown", "age_seconds": None, "max_age_seconds": max_age}
    try:
        observed = datetime.fromisoformat(observed_at)
        if observed.tzinfo is None or now.tzinfo is None:
            return result
        age = (now - observed).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return result
    result.update(age_seconds=round(age, 3),
                  status="future" if age < -2 else "stale" if age > max_age else "recent")
    return result


def write_private_json(path, value, *, overwrite=True):
    """Publish a complete 0600 JSON file; don't follow a predictable temp symlink."""
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(data)
        if overwrite:
            temporary.replace(path)
        else:
            # Exclusive creation: an existing human entry must never be overwritten.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_catalog():
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    return catalog["meta"], {c["id"]: c for c in catalog["cards"]}
