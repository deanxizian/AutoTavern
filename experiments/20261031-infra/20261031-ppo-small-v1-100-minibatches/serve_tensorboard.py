"""Serve logs with a bounded port probe on hosts that drop closed-port traffic."""

import argparse
import json
import os
from pathlib import Path
import socket
import threading

from tensorboard import program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--logdir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    board = program.TensorBoard()
    board.configure(argv=["tensorboard", "--logdir", str(args.logdir), "--host", "127.0.0.1",
                          "--port", str(args.port), "--reload_interval", "5", "--load_fast=false"])
    # TensorBoard's WerkzeugServer.connect_ex probe has no timeout by default.
    # Only bound startup probing; restore ordinary socket behavior after launch.
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(1.0)
    try:
        url = board.launch()
    finally:
        socket.setdefaulttimeout(previous)
    metadata = args.logdir / "tensorboard" / "process.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps({"pid": os.getpid(), "url": url, "logdir": str(args.logdir)}, indent=2) + "\n")
    print(f"TensorBoard ready: {url} (PID {os.getpid()})", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
