"""Authenticated loopback bridge for the native AutoTavern companion window."""

import argparse
from datetime import datetime, timezone
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import subprocess
import tempfile
import threading
import sys

# Direct launcher execution and `python -m` share the same package imports.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __package__:
    from .client_audit import discover_apps
    from .client_common import write_private_json
    from .window_session import LogFeed, WindowError, WindowSession
else:
    from client_audit import discover_apps
    from client_common import write_private_json
    from window_session import LogFeed, WindowError, WindowSession


ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 4 * 1024 * 1024


def make_server(session, token, port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never echo requests, credentials or private input.

        def reply(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A cancelled UI poll needs no second response.

        def authorized(self):
            expected_host = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected_host or self.headers.get("Origin") is not None:
                self.reply(403, {"error": "forbidden_origin", "message": "仅允许本机助手连接。"})
                return False
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.reply(401, {"error": "unauthorized", "message": "请从项目启动助手窗口。"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path != "/api/state":
                self.reply(404, {"error": "not_found"})
                return
            try:
                self.reply(200, session.state())
            except Exception:
                self.reply(503, {"error": "read_failed", "message": "暂时无法读取局面，请重新读取。"})

        def do_POST(self):
            if not self.authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY or self.headers.get_content_type() != "application/json":
                    self.reply(400, {"error": "invalid_body", "message": "需要有效的局面请求。"})
                    return
                def invalid_constant(_):
                    raise ValueError("Non-finite JSON")
                payload = json.loads(self.rfile.read(length), parse_constant=invalid_constant)
                if not isinstance(payload, dict):
                    raise ValueError("Expected object")
                if self.path == "/api/mode":
                    result = session.set_mode(payload.get("mode"))
                elif self.path == "/api/lobby-races":
                    result = session.set_lobby_races(payload.get("game_key"), payload.get("races"), payload.get("source"))
                elif self.path == "/api/feedback":
                    result = session.feedback(payload.get("proposal_id"), payload.get("outcome"))
                elif self.path == "/api/next-frame":
                    result = session.next_frame()
                elif self.path == "/api/preview":
                    result = session.publish_preview(payload)
                elif self.path == "/api/refresh":
                    result = session.state(force=True)
                else:
                    self.reply(404, {"error": "not_found"})
                    return
                self.reply(200, result)
            except WindowError as error:
                self.reply(409, {"error": error.code, "message": str(error)})
            except (ValueError, UnicodeError, TypeError):
                self.reply(400, {"error": "invalid_record", "message": "局面格式无效，未生成指引。"})
            except Exception:
                self.reply(503, {"error": "request_failed", "message": "请求未完成，请重新读取局面。"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-app", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--bundle", type=Path, default=ROOT / "runs/client-window/AutoTavernAssistant.app")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--model-config", type=Path, default=ROOT / "runs/client-model/selected.json")
    args = parser.parse_args()
    if not args.bundle.is_dir():
        parser.error("Build the app with ./script/build_and_run.sh first")
    apps = [args.client_app] if args.client_app else discover_apps()
    app = apps[0] if len(apps) == 1 else None
    logs = args.logs or (app.parent / "Logs" if app else None)
    run = ROOT / "runs/client-window/sessions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3))
    model_bridge = None
    if args.model_config.is_file():
        from tools.client_model_bridge import ModelBridge
        model_bridge = ModelBridge(json.loads(args.model_config.read_text()))
    from tools.client_races import RaceReader
    from tools.client_memory import PublicStateReader
    helper = args.bundle / "Contents/MacOS/AutoTavernMirrorReader"
    race_reader = RaceReader(helper, diagnostic_file=run / "races.jsonl")
    memory_reader = PublicStateReader(helper, diagnostic_file=run / "memory.jsonl")
    session = WindowSession(LogFeed(logs, app), receipt_file=run / "feedback.jsonl",
                            model_bridge=model_bridge, race_reader=race_reader, memory_reader=memory_reader)
    token = secrets.token_urlsafe(32)
    server = make_server(session, token)
    connection = run / "connection.json"
    connection_info = {"url": f"http://127.0.0.1:{server.server_port}", "token": token}
    write_private_json(connection, connection_info)
    # The GUI only needs an ephemeral loopback credential, not access to the repo
    # volume. Reading that volume during App initialization can block macOS open().
    native_session = tempfile.TemporaryDirectory(prefix="autotavern-assistant-")
    native_connection = Path(native_session.name) / "connection.json"
    write_private_json(native_connection, connection_info)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(json.dumps({"status": "window_bridge_started", "connection_file": str(connection)}, ensure_ascii=False), flush=True)
    child = None
    try:
        child = subprocess.Popen(["/usr/bin/open", "-n", "-W", str(args.bundle), "--args", "--connection", str(native_connection)])
        if args.verify:
            print("App bundle launch requested; bridge remains active until the window closes.", flush=True)
        return child.wait()
    except KeyboardInterrupt:
        # Stop only the companion app, never Hearthstone or a training process.
        subprocess.run(["pkill", "-x", "AutoTavernAssistant"], check=False)
        if child:
            child.wait(timeout=5)
        return 0
    finally:
        server.shutdown()
        server.server_close()
        connection.unlink(missing_ok=True)
        native_session.cleanup()
        if model_bridge:
            model_bridge.close()
        race_reader.close()
        memory_reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
