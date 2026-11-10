# V4：全动作循环惩罚与超限判负

日期：2026-10-31（北京时间）。状态：**prepared，未启动训练**。

本目录 [config.json](config.json) 是 V4 独立配置。三阶段计划与风险检查见
[V4–V6 计划](../V4-V6-PLAN.md)。此前 200 PPO、尚无投降的准备配置保存在
[preparation-history](preparation-history/20261031-before-100ppo-cycle-forfeit/)，没有改写 V1–V3。

## 规则

- 原名次效用 `(9 - 2 * placement) / 7` 保留。
- `operation_cost=0`、`cycle_cost=0.0002`、`cycle_detection="all_actions"`。
- `cycle_forfeit_repetitions=3`：连续三次完整无进展往返触发真实淘汰。
  用户确认的 `ABABAB` 表示 A、B 两个动作执行三组，第六个动作后判负。
  出牌／选择／取消等三步循环同样在第三组结束时触发。
- `turn_action_forfeit_limit=60`：前 60 个招募回合动作允许执行；若仍未结束，
  发起第 61 个动作时先判负，拦截旧规则的强制 End，不执行该动作或进入战斗。
  该超限尝试是终局策略转移，不冒充一次正常游戏操作；60 步内主动 End 不触发。
- 循环成本按每次可见状态返回收取，循环判负按完整闭环计数，两者不混用。
- End、真实跨回合、reset、观测不连续清空循环历史；遇到本回合未见的新可见状态时
  清零连续闭环数。同回合跨 rollout 保留。操作预算计数不因新状态而清零。
- reward 模块作判断，外层环境执行真实淘汰，正确处理卡池归还、淘汰效果、名次和终局。
  投降不是网络可选的新动作；通用游戏默认规则不变。
- 8 人存活判负为第 8；4 人存活判负为第 4。只发一次名次奖励。
  循环判负保留触发动作成本；操作超限不收取未执行动作的循环成本。
  `done=True`、非时间截断，PPO 不向下一局 bootstrap。
- 规则只施加到本实验学习者；冻结 V1／V2 对手保留旧行为。旧配置缺省关闭两种判负。

## 训练与评估

- 从 V3 最终保留的 PPO150 best 开始，继承 Adam；路径与 SHA-256 已绑定。
  来源 Adam 计数 5,600，本阶段最多新增 1,600。实际 best 的计数以选中检查点为准。
- 100 PPO；128 环境 × 1,024 步，共 13,107,200 个学习者样本。
- minibatch 8,192，16 个／PPO；microbatch 1,024，累积 8 次，epoch=1。
- LR 全程固定为 `1e-5`，无 warmup、无衰减；继承 Adam moments 和历史计数。
  配置复用已有 linear 调度，将起止值均设为 `1e-5`。
- 网络、BF16 前向／FP32 权重与 Adam、其余 PPO 参数沿用 V3。
- 对手：2 固定 V1 + 3 固定 V2 + 2 脚本；固定模型随机采样，每局轮换物理席位。
- 验证：同一混合对手，学习者随机／贪心各 64 局，PPO 0／50／100，不额外跑七脚本组。
- best 仅在混合随机组平均名次严格降低时更新；起点参与，相等保留较早检查点。
- 独立评估：同一混合对手，initial／best／final 各随机／贪心各 128 局，不参与选模。
  初始 V3 权重在 V4 新规则下重评，旧 V3 历史结果仍按原协议解释。
- 配置、阈值、指标记录到 W&B 项目 `eric_eric_/AutoTavern` 和本地。
  分开记录两种判负的比例、触发回合与结算名次，观察后期早退倾向。

模型目录仍为 `runs/20261031-ppo-v4-cycle-only/`，本次准备没有创建模型或在线 run。
`source-lock-verification.json` 和 `preparation-verification.json` 是旧准备快照，
终止规则验证见 `termination-preparation-verification.json`；
固定 LR 调整后的配置摘要与检查见 `fixed-learning-rate-verification.json`，旧验证快照不改写。
