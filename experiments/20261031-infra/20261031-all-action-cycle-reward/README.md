# 全动作循环奖励

2026-10-31：针对逐动作诊断发现的出牌／选择／取消循环，扩展训练奖励覆盖范围。
这是基础设施修改，未启动新的正式训练，也未修改实验 03 的进程、配置或已记录结果。

## 首次实现的配置（历史）

```json
"reward_shaping": {
  "operation_cost": 0.0001,
  "cycle_cost": 0.0002,
  "cycle_detection": "all_actions"
}
```

首次实现时，新初始化模板 `configs/ppo-small.json` 使用该配置。下一阶段从旧模型续训时需显式合入，
通用训练入口也支持 `--cycle-detection all_actions`。两个成本系数保持不变。

- 同一游戏回合内，所有非 End 动作返回已见完整可见状态时扣 `0.0002`。
- 覆盖移动往返、冻结往返、出牌→取消、出牌→选择→取消与技能菜单循环。
- 首次打开新菜单不罚；首次取消返回起点时计一次，后续反复进入已见菜单也计数。
- 正常施法、买卖、选择造成不同状态时，不会仅因动作类型重复扣分。
- Move／Freeze 的操作成本仍为 `0.0001`；出牌、选择、取消不收额外操作成本。
- End、终局、显式 reset、跨回合或观测不连续时清空历史；同回合跨 rollout 保留历史。
- 不增设 60 步触顶惩罚，不修改原生合法动作、观测、推理或模型结构。

检测只使用 native features/cards，忽略掩码，不访问隐藏计数器和 RNG。
可见重复不能证明完整模拟器状态相同，无法把所有未重复操作认定为有收益。

## 历史隔离与指标

`cycle_detection` 缺省为 `reversible_segment`，沿用此前连续 Move／Freeze 片段检测。
当前实验 03 的 `config.json` 与 `resolved-config.json` 均未添加新字段，解析仍为旧模式；
即使其后处理进程从更新后的源码启动，也保持旧奖励口径。
只读检查与两个配置的 SHA-256 记录在 `verification.json`，没有改写历史配置。

奖励仍经环境进入采样缓存、GAE、returns 与 PPO。
W&B 沿用 `train/cycle_returns`、`train/cycle_penalty` 及对应 `val/*`；
检测模式随奖励配置写入 resolved-config、检查点与验证结果。
新模式的循环次数与旧模式覆盖范围不同，比较实验时必须注明。

## 验证

`.venv/bin/python -m pytest -q`：**128 passed，38.72 秒**，包括 CUDA BF16 与离线 W&B 集成。
真实模拟器用 Tavern Dish Banana 验证两步循环、Forest's Bounty 验证三步循环，
同时检查成功施法、状态进展、原生批量封装、跨回合／episode 隔离、
旧模式兼容、PPO GAE／回报及 train／val 配置保存。

第一次受限沙箱执行时，39 项通过、2 项 W&B 测试因本地套接字限制未通过；
在允许本地套接字的环境完成上述完整检查，全部通过。
这是正确性验证，尚无使用新奖励训练后的对战效果结论。

实现：[rewards.py](../../../python/taverngym/learning/rewards.py)；
回归：[test_reward_shaping.py](../../../tests/test_reward_shaping.py)；
依据：[逐动作重复诊断](../20261031-turn-repetition-audit/README.md)。

## 后续确认：只罚循环（2026-10-31）

用户确认普通操作不收成本，只惩罚完整可见状态重复。新模板现在采用：

```json
"reward_shaping": {
  "operation_cost": 0,
  "cycle_cost": 0.0002,
  "cycle_detection": "all_actions"
}
```

普通移动、冻结、出牌、选择、技能操作不因执行本身扣分；在同回合返回已见状态时，
额外扣 `0.0002`，涵盖移动／冻结、出牌→取消、出牌→选择→取消、技能菜单往返。
名次奖励保留，不增加 60 步触顶惩罚或重复触发的强制结束。
`operation_penalty` 指标继续记录，采用新配置时应为 0。

当前实验 03 从 PPO177 恢复的训练仍沿用原成本与 `reversible_segment`，
原始配置 SHA-256 不变；新默认值只用于后续新训练。
历史 `verification.json` 保留首次完整检查结果；新增配置与验证见
`cycle-only-settings.json`、`cycle-only-verification.json`。

定向回归：`tests/test_reward_shaping.py` **33 passed，3.04 秒**。
覆盖零操作成本下的移动／冻结往返、出牌／选择／取消循环、原生批量封装，
并新增 Galakrond's Greed 技能打开目标选择→取消→再次打开的真实模拟器案例。
