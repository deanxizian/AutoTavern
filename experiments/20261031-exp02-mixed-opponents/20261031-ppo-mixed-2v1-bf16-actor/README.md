# 实验 02：2 个固定 V1、BF16 与独立 Actor 采样

2026-10-31。继承第一轮 best 网络，重置 Adam 与阶段计数，从头执行本阶段 100 次 PPO。
这次运行与先前停止的 4 V1 + 3 脚本尝试分开记录，旧现场及检查点保留。

在线指标：[W&B tgv920ha](https://forge.coreweave.com/wandb/eric_eric_/AutoTavern/runs/tgv920ha)。
实时状态见本地 `progress.json` 和 `experiment.json`；提交中的运行指标只是提交时的快照。
W&B 在线 train／val 标量已开启；Weave 初始化发生 TLS 错误，未影响 W&B 标量采集。

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
