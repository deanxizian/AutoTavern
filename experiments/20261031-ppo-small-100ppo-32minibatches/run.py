"""Run training, then validate and retain only the best model automatically."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from taverngym.learning.experiment import now, write_json


def main():
    folder = Path(__file__).resolve().parent
    repository = folder.parent.parent
    models = repository / "runs" / folder.name
    if models.exists():
        raise FileExistsError("Refusing to restart or overwrite this run")
    metadata = json.loads((folder / "experiment.json").read_text())
    metadata.update(status="starting", started_at=now(), training_started=True, supervisor_pid=os.getpid())
    write_json(folder / "experiment.json", metadata)
    command = [sys.executable, "-u", "-m", "taverngym.learning.experiment",
               "--config", str(folder / "config.json"),
               "--model", str(repository / "runs/20261031-ppo-small-init/initial_model.zip"),
               "--experiment-dir", str(folder), "--run-dir", str(models), "--device", "cuda"]
    with (folder / "console.txt").open("x") as stream:
        child = subprocess.Popen(command, cwd=repository, stdout=stream, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(folder / "launcher.json", {"supervisor_pid": os.getpid(), "training_pid": child.pid,
                                              "command": command, "started_at": now()})
        metadata.update(status="running", training_pid=child.pid, models_created=True)
        write_json(folder / "experiment.json", metadata)

        def stop(signum, frame):
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
    result = subprocess.run([sys.executable, "-u", str(folder / "finalize.py")], cwd=repository)
    if result.returncode:
        metadata.update(status="postprocessing_failed", exited_at=now(), exit_code=result.returncode)
        write_json(folder / "experiment.json", metadata)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
