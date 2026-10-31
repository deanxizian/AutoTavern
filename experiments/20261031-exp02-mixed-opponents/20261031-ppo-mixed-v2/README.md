# 实验 02：V1 继续训练，混合对手与动作成本

执行状态见 `experiment.json`、`progress.json`，W&B 链接见 `wandb.json`。
配置见 [config.json](config.json)，启动入口 [run.py](run.py)，完成后自动执行 [finalize.py](finalize.py)。
模型目录为 `runs/20261031-ppo-mixed-v2/`，Git 忽略全部权重。

学习者继承实验 01 的 `20261031-131146-best_model.zip` 网络与冻结文本向量；
清空旧 Adam 动量和阶段计数。源模型 SHA256 为
`7173d82343fcc79742838c72725c582966a3c5b7c84e15014debea292f32de24`，不会改写或删除。
4 个对手席位共享一份固定 V1 策略，使用随机采样；另外 3 个席位是原生脚本机器人。
每局重新打乱 8 个位置，角色 0 始终对应唯一学习者。
只有学习者的实际操作进入 PPO；对手动作与等待时间不构造额外样本。
学习者淘汰后名次已确定，立即重置该局，无须为不采集经验的旧模型对手模拟余局。
其他游戏等待时，已经可决策的学习者及整局状态保持不动。

| 配置 | 数值 |
| --- | --- |
| 网络 | d=192，4 层，8 头，3,491,978 个可训练参数 |
| 预算 | 100 次 PPO，26,214,400 条学习者新样本 |
| 每轮采样 | 256 环境 × 1,024 步 |
| 优化 | 32 个 minibatch × 8,192 samples，epoch=1；最多 3,200 次 Adam |
| microbatch | 1,024，累积 8 次 |
| 学习率 | 每个阶段重新线性 3e-4 → 3e-6；无 warmup |
| PPO | γ=1，λ=0.995，clip=0.2，target KL=0.05，梯度范数上限 1 |
| 损失系数 | entropy=0.002，value=0.5，placement auxiliary=0.1 |
| 奖励 | 原始名次效用 − 0.0001×Move/Freeze 次数 − 0.0002×无进展状态返回次数 |
| 动作上限 | 每招募回合 60 个动作，随后仅 EndTurn；不接入推理循环筛选器 |
| 精度 | FP32 参数／梯度／Adam，允许 TF32；不启用 BF16 autocast |
| 种子 | 训练 43，固定验证 20,000,002，独立测试 30,000,002 |

训练前 step=0 与 PPO 25/50/75/100，各评估四个条件、每条件 64 局：
混合对手随机／贪心、全脚本随机／贪心，共 1,280 局固定验证。
只有**混合对手、随机采样的平均名次**决定 best，相同分数保留较早版本。
独立测试每条件 128 局，对比 V1 起点、选定 best 与 final；best=final 时复用同一结果，
不把它算成额外独立样本。测试与 TavernBench 均不参与选模。

W&B 使用 `eric_eric_/AutoTavern`，同时调用 `weave.init`，仅上传标量与配置。
每次实际 Adam 记录 train loss／KL／LR；每次 PPO 记录采样对局名次、成本、循环与操作数。
主验证指标为 `val/mean_placement`、`val/operation_penalty`、`val/cycle_penalty` 等；
另外三组分别加 `greedy_`、`scripted_`、`scripted_greedy_` 前缀，保持一层 `/`。
`cap_end_rate` 为每局达到动作上限后被迫 EndTurn 的招募回合比例，再按已完成整局平均。

训练期间保留起点、latest、每 25 轮 checkpoint、best 与 final 便于复核。
独立测试后只留下 `YYYYMMDD-HHMMSS-best_model.zip`，保留清理清单和所有指标。
随后自动运行 [TavernBench](../20261031-ppo-mixed-v2-tavernbench/README.md)，用相同适配器比较 V1 与 V2。

启动前验证见 [GPU 预跑](../../20261031-infra/20261031-mixed-opponent-preflight/README.md)。
整体日期目录与阶段计数独立于第一次实验，不会续写第一轮的 W&B run。
