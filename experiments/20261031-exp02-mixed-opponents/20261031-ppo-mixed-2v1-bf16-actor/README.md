# 实验 02：2 个固定 V1、BF16 与独立 Actor 采样

2026-10-31。继承第一轮 best 网络，重置 Adam 与阶段计数，从头执行本阶段 100 次 PPO。
这次运行与先前停止的 4 V1 + 3 脚本尝试分开记录，旧现场及检查点保留。

在线指标：[W&B tgv920ha](https://forge.coreweave.com/wandb/eric_eric_/AutoTavern/runs/tgv920ha)。
实时状态见本地 `progress.json` 和 `experiment.json`；提交中的运行指标只是提交时的快照。
W&B 在线 train／val 标量已开启；初次启动的 Weave TLS 错误未影响 W&B 标量采集，
恢复训练时 Weave 已成功初始化。

17:20 在第 23 轮采样时，较早席位死亡触发大猫发现，导致较晚席位的旧动作失效。
原始失败现场已归档；修复后于 17:42 从第 22 轮完整检查点续训，保留 Adam 和学习率进度，
复用同一 W&B run。复现、回归与恢复说明见
[批量动作失效修复](../../20261031-infra/20261031-batch-action-deferral/README.md)。

100 次 PPO、3,200 次 Adam、26,214,400 个学习者样本已完成；北京时间 19:12 完成独立评估、
TavernBench 与 best-only 留存。固定验证主条件为 **2 个固定 V1 + 5 个脚本，学习者随机采样，每点 64 局**：

| PPO 迭代 | 0（V1 起点） | 25 | 50 | 75 | 100（最终 best） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 平均名次 | 2.5625 | 2.90625 | 2.671875 | 2.703125 | **2.3125** |

中间验证有退化，最终第 100 轮在预设选模条件下超过起点；不能只根据前几次验证断言训练失败。
以下为独立随机种子评估，各行、各版本 **128 局**；起点使用原 V1，V2 使用第 100 轮 best。
两版均 BF16 前向，混合对手中的固定 V1 均随机采样；贪心只改变被测模型。

| 对手与被测模型选动作方式 | V1 平均名次 | V2 平均名次 | V1 吃鸡率 | V2 吃鸡率 |
| --- | ---: | ---: | ---: | ---: |
| 2 固定 V1 + 5 脚本，随机采样 | 2.6797 | **2.1641** | 25.78% | **42.97%** |
| 2 固定 V1 + 5 脚本，贪心 | 4.0313 | **2.9297** | 18.75% | **36.72%** |
| 7 脚本，随机采样 | 1.9375 | **1.6406** | 47.66% | **67.19%** |
| 7 脚本，贪心 | 3.6953 | **2.4453** | 29.69% | **48.44%** |

完整结果见 [results.json](results.json)，起点与 best 的独立评估见
[evaluation-initial-heldout.json](evaluation-initial-heldout.json)、
[evaluation-best-heldout.json](evaluation-best-heldout.json)。这些结果不参与选模。
操作数、循环返回数、cap_end_rate 仍需分别判断；没有无奖励惩罚对照，不能归因于某一项惩罚。

最终只保留本地 `runs/20261031-ppo-mixed-2v1-bf16-actor/20261031-165718-best_model.zip`，
SHA-256 为 `e4a748e933af5b642c50c223502d127ed1742599d841ab7d3838cf26df7ea6b4`；
ZIP 为 53,924,286 字节，权重 FP32，可使用 BF16 前向。留存清单见 [retention.json](retention.json)。
后续同局对战见 [2 脚本 + 3 V1 + 3 V2](../20261031-match-2script-3v1-3v2/README.md)。

| 项目 | 设置 |
| --- | --- |
| 对局 | 5 脚本 + 2 固定 V1 + 1 学习者；每局角色重新随机分配席位 |
| 初始化 | 第一轮 `20261031-131146-best_model.zip`；清空 Adam、本阶段 PPO 和样本计数 |
| 网络 | d=192，4 层，8 头，3,491,978 个可训练参数 |
| 采样 | 独立 Actor 调度，跨局分别合批学习者和固定 V1；固定角色共用一份权重 |
| 预算 | 100 PPO × 256 局 × 1,024 个实际学习者决策 = 26,214,400 个新样本 |
| 更新 | 每 PPO 32 个有效 minibatch × 8,192，epoch=1；最多 3,200 次 Adam |
| 梯度累积 | microbatch=1,024，累积 8 次 |
| 精度 | 三类前向均 BF16；输出概率、GAE、loss、权重、梯度、Adam 和保存权重 FP32 |
| 学习率 | 3e-4 → 3e-6，按 PPO 采样进度线性下降 |
| PPO | gamma=1，lambda=0.995，clip=0.2，target KL=0.05，梯度裁剪=1 |
| 损失 | entropy=0.002，value=0.5，placement auxiliary=0.1 |
| 奖励 | 原始名次效用 − 0.0001×Move/Freeze − 0.0002×无进展状态返回 |
| 动作上限 | 每个招募回合 60；不启用推理动作过滤 |
| rollout | 无损压缩 RAM，约 0.82 GiB 缓存；不包含进程和模拟器内存 |

每局按自己的决策序号写入 `[step, game]` 缓存；只有真实学习者操作进入 PPO。
终局奖励只结算一次，名次标签仅回填本 rollout 的已结束片段，GAE 不跨 episode 传播。
所有局均达到 1,024 个学习者决策后才更新权重，采样期间固定 V1 和学习者参数都保持不变。
各局间没有每个学习者动作的全局等待；最后的 rollout 配额仍需等待最慢的局。

训练前及 PPO 25／50／75／100 做四条件验证，各 64 局：混合随机／混合贪心／全脚本随机／全脚本贪心。
混合随机平均名次决定 best，相同分数保留更早版本。起点与后续评估共享 `val/*`，起点在 step=0。
每次 Adam 写 train loss／KL／LR；每 PPO 写采样名次、操作与循环成本、循环次数、cap_end_rate、
Actor batch 大小及耗时。指标保持一层 `/`；未完成整局不计入每局统计均值。

结束后以独立种子、每条件 128 局比较起点／best／final，best=final 时复用同一结果。
再跑 [TavernBench](../20261031-ppo-mixed-2v1-bf16-actor-tavernbench/README.md)，完成后只保留带北京时间前缀的 best。
源 V1 不会删除，权重只放本地 `runs/20261031-ppo-mixed-2v1-bf16-actor/`，不提交 Git 或上传 W&B。

正式启动前 99 项 Python 测试通过，包含 CUDA 与离线 W&B。
真实首轮学习率 2.9703e-4 的一次性预检完成全部 262,144 个样本和 32 次 Adam：
采样 41.56 秒、更新 17.25 秒、峰值 allocated 3.75 GiB；V1 文件及固定策略未改变。
这是一次性预检，不计入正式 100 轮。详情见 [接入记录](../../20261031-infra/20261031-actor-ppo-integration/README.md)。

配置：[config.json](config.json)；来源及测试：[provenance.json](provenance.json)。
监督入口 `run.py` 会在训练完成后调用 `finalize.py` 完成测试、benchmark 与 best 留存；训练失败时保留现场。
