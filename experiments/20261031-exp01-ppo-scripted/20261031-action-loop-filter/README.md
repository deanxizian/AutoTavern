# 实验 01：可逆动作循环筛选对照

日期：2026-10-31，Asia/Shanghai。状态：**完成**。使用原有第 100 轮权重，仅在推理选动作时增加可选筛选。
未修改权重、奖励或原生合法动作规则，未启动第二轮训练。

## 规则

实现：[ReversibleActionFilter](../../../python/taverngym/learning/action_filter.py)。

- 记录当前没有实质变化的局面中，已经实际到达过的阵型和酒馆冻结状态。
- 移动若会回到已见阵型，或冻结按钮若会回到已见冻结状态，就从这次模型可选动作中移除。
- 金币、手牌、商店内容、卡牌属性、待选项、回合等发生变化，或执行购买／打牌／刷新等其他操作后，
  重置这一段历史，允许新的站位与冻结决策。
- 对连续的移动／冻结另设 **8 次预算**，避免一直尝试不同排列。该值是本地初始选择，未按评测成绩调参。
- 保留原生合法的购买、出售、打牌、选择、刷新和结束回合等动作，始终只取原生合法性掩码的子集。
- 只记录实际执行的动作；反复查看同一个建议不会消耗预算。每个玩家／对局使用独立实例。

该筛选属于策略约束，不把被过滤的动作宣称为游戏规则上的非法动作。
原生 `NativeVectorEnv`、`EntityPolicy`、`model.predict()` 的默认行为均未改变；当前实验使用独立适配器。

## 128 局对照

对照使用与 [原始诊断](../20261031-action-loop-diagnostic/README.md) 相同的模型、种子 50000001、
32 房间、每房间两局配额、七个脚本对手和每回合 60 次操作上限。
计数器直接复用原始诊断代码，包含同回合内重叠的重复状态访问，跨回合／实质动作后不串联。

| 模式 | 条件 | 有换位回退的对局 | 有冻结往返的对局 | 平均名次 ↓ | 总动作数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 确定性 argmax | 原模型直接推理 | 64/64 | 63/64 | 3.65625 | 48,177 |
| 确定性 argmax | **加筛选** | **0/64** | **0/64** | **1.765625** | **31,840** |
| 概率采样 | 原模型直接推理 | 64/64 | 64/64 | 1.71875 | 49,215 |
| 概率采样 | **加筛选** | **0/64** | **0/64** | **2.03125** | **37,552** |

确定性模式动作数下降约 33.91%，受测循环消失，名次改善；采样模式动作数下降约 23.70%，
但本次名次有所回退。以上是各 64 局的描述性对照，尚未验证更多随机种子的稳定性。
两个模式都仍可能执行较多不重复的移动；最长连续可逆操作由 60 降至 8，不等于已学会最少点击。

这是**旧权重加推理规则**的结果，不能称为权重本身已经学会不循环。
单独复制 `.zip` 不会启用筛选。若最终部署保留此约束，训练、评估和实战应使用一致规则；
PPO 采样和更新必须使用同一份实际筛选掩码，不能采样后替换动作而继续记录原动作概率。
现有 MaskablePPO 的 mask 接口可承载该子集，参考 [官方文档](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)。

如希望只靠裸模型推理消除循环，还需要另做包含操作效率／重复状态代价的训练对照，并重新测裸策略循环率。
本次未更改奖励、gamma、网络输入或训练配方。

## 调用方式

```python
from taverngym.learning.action_filter import ReversibleActionFilter

guard = ReversibleActionFilter(max_reversible_actions=8)  # 每个玩家各一份
mask = guard.mask(obs)
action, _ = model.predict(obs, deterministic=True, action_masks=mask)
# 实际执行后拿到新观测；仅展示建议时不要调用 commit。
guard.commit(int(action), obs, next_obs, done=episode_done)
```

外部用户仍可手动操作；改变决策流程或重开游戏时可调用 `guard.reset()`。
当前工具只支持原生观测格式，其他实战项目应在自己的推理入口显式接入，并确认动作和观测定义一致。

## 验证与复现

五项原生游戏测试覆盖：立即逆向移动、更长移动循环、冻结往返、刷新后重置、
冻结不丢失移动历史、预算与新对局重置、输入不变及不扩大原生合法动作集合。
对照中 128 局均完成，原始计数器未发现受测循环，模型文件摘要与 Adam 步数保持不变。

```bash
uv run --locked --group train --extra test python -m pytest -q tests/test_action_filter.py
uv run --locked --group train python experiments/20261031-exp01-ppo-scripted/20261031-action-loop-filter/run.py
```

产物：[results.json](results.json)、[确定性结果](greedy-results.json)、[采样结果](sampled-results.json)、
[确定性轨迹](greedy-actions.jsonl.gz)、[采样轨迹](sampled-actions.jsonl.gz)。
主脚本拒绝覆盖已有完整结果。第二轮正式训练仍以单独实验配置为准。

前后对照标量已同步 [W&B](https://forge.coreweave.com/wandb/eric_eric_/AutoTavern/runs/fnfp39ez)，没有上传权重或存档。完整 Python 回归测试为 67 passed。
