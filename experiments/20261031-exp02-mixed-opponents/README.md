# 实验 02：混合对手与动作成本

第二次正式模型实验的记录入口。
后续在这里新建 `yyyymmdd-{exp-name}/`，保存训练配置、结果及对应 benchmark；模型放到 runs。

**当前正式运行：** [2 V1 + BF16 + 独立 Actor](20261031-ppo-mixed-2v1-bf16-actor/README.md)，
100 次 PPO，从第一轮 V1 起点重新初始化本阶段 Adam 与计数。
[W&B tgv920ha](https://forge.coreweave.com/wandb/eric_eric_/AutoTavern/runs/tgv920ha)。
当前设置见 [next-settings.json](next-settings.json)：5 脚本 + 2 固定 V1 + 1 学习者，
三类前向 BF16，训练状态与权重 FP32，独立 Actor 已接入正式 PPO。

先前 4 模型 + 3 脚本尝试保留为停止的历史记录：完成 2 PPO / 64 Adam 后，
第三轮采样发生动作拒绝；用户当时要求停止。新运行与该历史目录及检查点分开，未从失败 rollout 续训。
新调度的完整一轮预检未复现该错误，仍严格校验并在错误时保存诊断。

之前试跑方案：1 个学习者从实验 01 的 best 初始化，对阵 4 个冻结的实验 01 模型席位与 3 个脚本席位。
已配置 [100 PPO 正式训练](20261031-ppo-mixed-v2/README.md)，
执行状态、W&B 链接与进度以该目录中的运行记录为准。
在线指标：[W&B run pczkdfc4](https://forge.coreweave.com/wandb/eric_eric_/AutoTavern/runs/pczkdfc4)。
训练完成后自动做独立整局测试、best-only 留存及 [TavernBench](20261031-ppo-mixed-v2-tavernbench/README.md)。

已确认将减少无效操作纳入训练目标，配置片段见 [reward-shaping.json](reward-shaping.json)：

| 奖励组成 | 数值 |
| --- | ---: |
| 原始终局名次奖励 | `(9 - 2 * placement) / 7` |
| 每次 Move / Freeze 的操作成本 | `−0.0001` |
| 同一无进展片段中返回已见完整可见状态的额外成本 | `−0.0002` |

因此通常的 `A → B → A` 两次操作分别为 `−0.0001`、`−0.0003`。
只看玩家可见 features/cards；不把合法掩码计入循环状态，避免操作计数改变掩码后漏判。
Move/Freeze 以外的操作、回合／公开资源／属性等变化、新对局和 reset 都重置循环历史。
检测完整可见状态，而不是仅检查站位；例如移动与冻结交错时，只有完整状态重复才加罚。

该奖励在采样时加入，进入 GAE、critic returns 和 PPO 策略目标；名次辅助标签保持真实名次。
不接入推理筛选器，不禁止逆向动作，不改网络输入或权重格式。验证使用裸策略，仍按平均名次选 best；
同时记录 `train/*`、`val/*` 的名次奖励、两项惩罚、总奖励、每局操作数和循环次数。
系数是本项目的初始实验选择，尚未证明能减少训练后模型的循环，需要同时复核贪心／随机采样的循环与胜率。

接入及验证记录见 [奖励训练基础设施](../20261031-infra/20261031-reward-shaping/README.md)。
