"""Nonblocking latest-frame bridge to the separate local model process."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import selectors
import threading

from .client_actions import frame_identity


ROOT = Path(__file__).resolve().parents[1]


class ModelBridge:
    def __init__(self, config, *, launch=None):
        self.config = dict(config)
        self.condition = threading.Condition()
        self.pending = None
        self.current_key = None
        self.result = {"status": "loading"}
        self.policy = None
        self.closed = False
        self.process = None
        self.failed = False
        self.launch = launch or subprocess.Popen
        self.thread = threading.Thread(target=self._run, name="client-policy", daemon=True)
        self.thread.start()

    def sample(self, snapshot, *, enabled=True):
        key = frame_identity(snapshot)
        actionable = enabled and snapshot.get("status") in ("snapshot", "incomplete_snapshot") and snapshot.get("game_status") != "complete"
        with self.condition:
            if self.failed:
                return {**deepcopy(self.result), "policy": deepcopy(self.policy)}
            if not actionable:
                self.current_key, self.pending = None, None
                self.result = {"status": "waiting_for_game"}
            elif key != self.current_key:
                self.current_key = key
                # IncrementalReader shares cached rows; take ownership at enqueue.
                self.pending = (key, deepcopy(snapshot))
                self.result = {"status": "converting", "frame_id": key}
                self.condition.notify()
            state = deepcopy(self.result)
            state["policy"] = deepcopy(self.policy)
            return state

    def _run(self):
        process = None
        try:
            python = str(self.config.get("python") or ROOT / "runs/model-env/bin/python")
            command = [python, "-m", "tools.client_model", "worker", "--checkpoint", str(self.config["checkpoint"]),
                       "--threads", str(self.config.get("threads", 4))]
            if self.config.get("sha256"):
                command += ["--sha256", self.config["sha256"]]
            if self.config.get("position_bundle"):
                command += ["--position-bundle", self.config["position_bundle"]]
            if self.config.get("position_sha256"):
                command += ["--position-sha256", self.config["position_sha256"]]
            process = self.launch(command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
            with self.condition:
                self.process = process
                if self.closed:
                    return
            def read_message(timeout):
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    if not selector.select(timeout):
                        raise TimeoutError("Model worker did not respond")
                return json.loads(process.stdout.readline())
            ready = read_message(30)
            if ready.get("status") != "loaded":
                raise ValueError("Policy did not load")
            with self.condition:
                self.policy = ready["policy"]
                if self.current_key is None:
                    self.result = {"status": "waiting_for_game"}
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.pending is not None)
                    if self.closed:
                        return
                    key, snapshot = self.pending
                    self.pending = None
                process.stdin.write(json.dumps({"snapshot": snapshot}, ensure_ascii=False, allow_nan=False) + "\n")
                process.stdin.flush()
                result = read_message(10)
                with self.condition:
                    # A changed frame, mode, session, stale source or completed game
                    # invalidates in-flight work; old output never reaches the UI.
                    if key == self.current_key and not self.closed:
                        if result.get("frame_id") != key:
                            self.result = {"status": "failed", "error_type": "FrameMismatch"}
                        else:
                            self.result = result
        except Exception as error:
            with self.condition:
                self.failed = True
                self.result = {"status": "failed", "error_type": type(error).__name__}
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

    def close(self):
        with self.condition:
            self.closed = True
            self.current_key = self.pending = None
            process = self.process
            self.condition.notify_all()
        if process is not None and process.poll() is None:
            process.terminate()
        self.thread.join(timeout=3)


def model_summary(state):
    """Product text; technical coverage stays in the state-review panel."""
    if not state:
        return "尚未选择模型", "等待接入训练模型。", []
    policy = state.get("policy") or {}
    name = f"u{policy['update']}" if policy.get("update") is not None else "训练模型"
    status = state.get("status")
    title = f"已载入 {name}" if policy else "模型载入中"
    if status == "converting": title = "正在更新建议"
    detail = {"loading": "正在载入本地模型。", "waiting_for_game": "等待新的对局局面。",
        "converting": "正在检查当前局面的模型输入。", "input_incomplete": "部分效果和成长数据尚未完成转换，暂不生成操作建议。",
        "inferred": "根据当前局面实时更新；请按窗口建议手动操作。", "failed": "模型连接未完成，请查看局面核对中的状态。"}.get(status, "等待模型状态。")
    blockers = []
    if status == "input_incomplete":
        gaps = [g["code"] for g in state.get("input", {}).get("gaps", [])]
        if "lobby_races_required" in gaps:
            title = "确认本局种族"
            detail = "请确认本局的 5 个随从种族，随后开始给出建议。"
        elif any(g in ("stale_options", "stale_choices", "not_action_phase") for g in gaps):
            title = "等待可操作局面"
            detail = "等待下一次可操作局面。"
        else:
            title = "建议暂不可用"
            labels = {"ambiguous_action": "当前操作存在未能区分的模式，已暂停这份局面的建议。", "hand_lock_unmapped": "当前手牌的锁定效果尚未支持。",
                "unstable_snapshot": "日志正在更新，完成后自动恢复建议。", "shop_price_unresolved": "当前酒馆价格尚未完整读到。"}
            detail = next((labels[g] for g in gaps if g in labels), "当前卡牌效果尚未完整支持；日志已保留用于回放排查。")
        blockers = [detail]
    if status == "inferred" and state.get("proposal"):
        detail += f" 本次推理 {state['proposal']['inference_ms']:.1f} 毫秒。"
    elif status == "failed":
        title = "模型连接未完成"
        blockers = ["模型载入或推理失败"]
    return title, detail, blockers
