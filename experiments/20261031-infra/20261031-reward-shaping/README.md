# 训练奖励：操作成本与无进展循环成本

2026-10-31 用户确认在训练目标中减少换位往返、冻结／解冻循环。
本记录为训练基础设施验证，未启动第二次正式训练，没有改写既有模型。

实现见 [rewards.py](../../../python/taverngym/learning/rewards.py)，
通过原生批量环境和 Gymnasium 训练环境返回 shaped reward，随后由原有 PPO 计算 GAE 与 returns。
新初始化模板与 [实验 02 配置片段](../../20261031-exp02-mixed-opponents/reward-shaping.json)
采用 `operation_cost=0.0001`、`cycle_cost=0.0002`。
旧配置缺少该字段或为 null 时保留原始终局奖励。

终局名次奖励保持 `(9 - 2 * placement) / 7`；每次 Move / Freeze 减去操作成本；
在没有其他有效操作和公开状态进展的连续片段中，返回完整已见可见状态时再减去循环成本。
仅比较 native features/cards，不读取 state_json、对手隐藏信息或模型预测。
state_json 只在验证脚本回放历史模拟器快照时使用，不参与奖励判断或网络输入。
重复状态键不包含 action_mask，因此动作／Freeze 次数改变合法性不会使循环漏判。
公开资源、属性、手牌、选项、回合等变化重置历史；非 Move/Freeze 动作与新对局也重置。

这是训练目标的扩展，操作依然合法，不调用推理筛选器，网络架构和保存格式不变。
目前仅验证奖励接入正确，尚无新训练结果，不能据此宣称模型已学会避免循环。

验证范围：真实模拟器中的交换往返、三步循环、冻结往返、交错移动／冻结、有效操作与
公开进展后的历史重置、跨环境／跨对局隔离、原生状态与掩码不变，
以及 shaped reward 进入 PPO buffer、GAE 和回报目标。
W&B 离线集成验证覆盖初始与周期 val 指标、成本配置保存及原有每 Adam 点数。
完整检查结果与历史案例回放见 [verification.json](verification.json)。

```bash
.venv/bin/python -m pytest -q
```

训练和验证指标详见 [训练指标文档](../../../docs/TRAINING_METRICS.md)。
正式训练后需要继续比较裸模型在贪心／随机模式下的循环、操作数、动作上限触发率与名次，
防止操作成本使模型过早结束回合。
