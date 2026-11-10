"""Supervise the new stage and run held-out evaluation and best-only retention."""

import json
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys

from taverngym.learning.experiment import now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Continue the latest complete PPO checkpoint")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    root = next(p for p in folder.parents if (p / "pyproject.toml").exists())
    models = root / "runs" / folder.name
    if models.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {models}")
    config = json.loads((folder / "config.json").read_text())
    metadata = json.loads((folder / "experiment.json").read_text())
    checkpoint = models / "latest_model.zip" if args.resume else root / config["initial_model"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    command = [sys.executable, "-u", "-m", "taverngym.learning.experiment",
               "--config", str(folder / "config.json"), "--model", str(checkpoint),
               "--experiment-dir", str(folder), "--run-dir", str(models), "--device", "cuda",
               "--resume" if args.resume else "--continue-stage"]
    stamp = now().replace(":", "").replace(".", "")
    if args.resume:
        history = folder / "resume-history"
        history.mkdir(exist_ok=True)
        write_json(history / f"{stamp}-supervisor.json", {"previous_execution": metadata,
                   "resumed_at": now(), "checkpoint": str(checkpoint),
                   "reason": "explicit continuation from the latest complete PPO checkpoint"})
        metadata.pop("exit_code", None)
        metadata.pop("exited_at", None)
        metadata["resumed_at"] = now()
    else:
        metadata["started_at"] = now()
    metadata.update(status="starting", training_started=True, supervisor_pid=os.getpid())
    write_json(folder / "experiment.json", metadata)
    console = folder / (f"console-resume-{stamp}.txt" if args.resume else "console.txt")
    with console.open("x", buffering=1) as stream:
        child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(folder / "launcher.json", {"supervisor_pid": os.getpid(), "training_pid": child.pid,
                                              "command": command, "started_at": now(), "console": str(console)})
        metadata.update(status="running", training_pid=child.pid)
        write_json(folder / "experiment.json", metadata)

        def stop(signum, frame):
            write_json(folder / "last-supervisor-signal.json", {
                "received_at": now(), "pid": os.getpid(), "signal": signal.Signals(signum).name,
                "signal_number": signum, "child_pid": child.pid,
            })
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        code = child.wait()
        progress_path = folder / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        if code != 0 or progress.get("status") != "completed":
            metadata.update(status=progress.get("status", "failed"), exited_at=now(), exit_code=code)
            write_json(folder / "experiment.json", metadata)
            return code or 1
        metadata.update(status="postprocessing", completed_training_at=now())
        write_json(folder / "experiment.json", metadata)
        code = subprocess.call([sys.executable, "-u", str(folder / "finalize.py")],
                               cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    if code:
        metadata = json.loads((folder / "experiment.json").read_text())
        metadata.update(status="postprocessing_failed", exit_code=code, exited_at=now())
        write_json(folder / "experiment.json", metadata)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
