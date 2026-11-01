# V4 → V5 → V6 固定学习率执行实例

**终态：2026-10-31 23:47 用户明确停止，旧实例不得恢复。** 原固定对手计划已被用户纠正；
实际仅 V4 完成 3 PPO、48 次新 Adam，累计采样 435,339（含未完成第 4 PPO 的 42,123）。
完整 latest 为 PPO3／393,216 样本／Adam 总计 5648。V5/V6 未启动。
所有模型、配置、W&B 与日志原样保留，见 `user-stop-final.json`。
新的 V4 必须从原 V3 PPO150 best／Adam5600 重起；新对手表待用户确认，尚未创建新实验配置。
下面是旧实例的历史执行协议和入口记录，不构成重新启动授权。

实例：`20261031-v456-fixedlr-01`。用户在主会话明确授权“开始”；总控先完成检查，
等待主 agent 确认独立 training_monitor 已就绪，再启动正式进程。

目标与假设：从 V3 PPO150 best 的完整网络与 Adam 状态继续，在固定对手与开发验证下，
观察循环成本和真实判负规则能否改善学习者平均名次。网络与训练核心复用已有验证实现；
本轮只补有限阶段执行、来源绑定、恢复边界、后处理和来源留存。

三个阶段各 100 PPO；128 环境 × 1024 步；minibatch 8192、microbatch 1024、epoch 1。
每阶段 LR 恒为 `1e-5`，继承所选来源的 Adam moments／历史计数，阶段计数归零。
总预算 300 PPO、39,321,600 个学习者样本、最多 4800 次新 Adam。
对手固定为 **2 个冻结 V1 + 3 个冻结 V2 + 2 个脚本**；模型对手随机采样。

学习者采用 operation cost 0、all_actions cycle cost 0.0002、三次完整循环判负、
第 61 次招募动作请求前判负。旧 V1/V2 对手不新增判负规则。
每阶段 PPO 0/50/100，学习者随机／贪心各 64 局；仅随机组均名次严格降低选 best，
相等保留较早版本。独立 initial/best/final 每种模式各 128 局，永不参与选模。
下一阶段继承前阶段 best，即使 best 是 PPO0；V6 独立评估与留存后结束。

正式入口（由总控独占执行）：

```bash
.venv/bin/python -u -m taverngym.learning.stage_chain \
  --manifest experiments/20261031-infra/20261031-training-agents/instances/20261031-v456-fixedlr-01/manifest.json
```

`manifest.json` 固定阶段、配置哈希、首来源、冻结对手、代码摘要与预算。
`controller-status.json` 原子记录阶段、PID、日志、完整检查点、预算、最后进展和停止原因。
`controller.lock` 是内核排他锁；启动时还逐阶段扫描精确进程参数，防止控制器退出后子进程仍活跃。
监控记录由 training_monitor 独占，总控不覆盖监控历史。

正常完成的阶段写 `results.json` 后，按 `retention.json` 清理本阶段非 best 模型；
来源与冻结对手不清理，清理中断可幂等续接。来源绑定先写持久意图，再更新配置，保留准备配置。

故障后先诊断并保留现场，再由总控用 `--resume --recovery-reason "具体诊断"` 接续，
每阶段最多一次。完整 PPO 已完成而验证/metrics 未结算时，专门补同协议验证与记录；
PPO100 后直接补独立评估，不多跑 PPO。无法还原的 rollout outcome/loss 明确标记，不造数。
若丢弃 partial 采样／Adam 会超出固定预算，停止推进并报告；不静默追加预算。
收到 SIGTERM/SIGINT 按明确停止处理，不自动恢复。已停止实例不得自动重启。

验证与采纳证据见 `preflight-verification.json`、各阶段 `source-checkpoint.json`、
`source-binding.json`、评估、W&B journal、`results.json` 和 `retention.json`。
本实例不修改通用历史池晋级门槛、不更新全局冠军、不部署。

停机后只读诊断使用原 V3、旧固定阵容，训练种子 44/45/46，学习者随机／贪心各 3 局，
6/6 真实第八名循环判负。实际循环包括六次 Freeze、三组 Play/Choose/Cancel、
三组 HeroPower/Cancel；完整可见状态哈希确实返回，动作均合法，新状态正确清零计数。
这六局没有检测／结算错误的证据，不能将首回合高判负率仅归因于缺少 V3 对手。
见 `initial-cycle-diagnosis.json`、`initial-cycle-traces.json` 与可复现的
`diagnose_initial_cycles.py`。诊断没有 Adam 更新、没有修改规则、没有读取独立 heldout。
