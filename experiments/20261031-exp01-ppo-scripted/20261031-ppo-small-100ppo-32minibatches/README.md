# 实验 01 训练：100 次 PPO × 32 个 minibatch，epoch=1

日期：2026-10-31，Asia/Shanghai。状态：**已完成**。
13:11:46 开始，14:08:41 完成训练，14:09:01 完成独立测试、核验与最佳模型留存。
共完成 100 次 PPO、3,200 次 Adam、26,214,400 条新样本；全流程约 57 分 15 秒。

## 最终结果

固定验证每次 64 局，种子 20000001；所有模型均随机采样合法动作，对阵七个相同策略的脚本机器人。

| PPO 迭代 | 平均名次 ↓ | 前四率 | 第一名率 |
| ---: | ---: | ---: | ---: |
| 0 | 7.921875 | 0% | 0% |
| 25 | 4.390625 | 50.0000% | 14.0625% |
| 50 | 3.015625 | 79.6875% | 20.3125% |
| 75 | 2.187500 | 95.3125% | 31.2500% |
| 100 | **1.812500** | **98.4375%** | **45.3125%** |

按固定验证平均名次选出的 best 为第 100 轮，与 final 权重一致。
另一种子 30000001 的 128 局用于独立复核，不参与选模：

| 模型 | 平均名次 ↓ | 前四率 | 第一名率 |
| --- | ---: | ---: | ---: |
| 初始模型 | 7.9765625 | 0% | 0% |
| best / final | **1.7656250** | **93.7500%（120/128）** | **58.59375%（75/128）** |

独立测试平均名次改善 6.2109375 名。best 和 final 的测试种子、权重与结果相同，
其重复执行不应当作 256 局独立测试样本。结果只代表此处的脚本对手协议。

训练计时 3,413.93 秒；其中采样累计 1,168.46 秒、更新累计 2,193.47 秒。
学习率、Adam 次数、有限权重及初始化源文件摘要均已核验。
已删除本实验其余 7 个模型文件，仅保留 `runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip`。
独立的零步初始化与其他实验不受影响。
W&B 已完成同步：3,200 个 train loss 点、5 个 val 点，以及 initial/best/final 的 test 指标。

详细记录：[results.json](results.json)、[verification.json](verification.json)、[retention.json](retention.json)。

用户指定总 PPO 迭代 100、minibatch 数量 32，并讨论了采集新数据和重复使用数据的取舍。
本配置保留原来已执行的 epoch=1；此前 epoch=2 是未执行的候选，没有将它默认为用户选择。
新配置单独记录，不改写旧 300 迭代实验或已完成的 100-minibatch 试跑。

## 预算

| 项目 | 设置 |
| --- | ---: |
| PPO iterations | **100** |
| 并行环境数 × 每环境采样步数 | 256 × 1,024 |
| 每次 PPO 新采样 B | **262,144 samples** |
| 每个有效 minibatch b | **8,192 samples** |
| 每个 epoch 的 minibatch 数 B/b | **32** |
| epochs E | **1** |
| 每次 PPO 的 Adam 更新上限 | **32** |
| 总 Adam 更新上限 | **3,200** |
| 总新采样量 | **26,214,400 samples** |
| microbatch／梯度累积 | 1,024／8 |
| 每次 PPO／总 microbatch 反向上限 | 256／25,600 |
| LR | 3e-4 → 3e-6，线性，无 warmup |
| 精度 | FP32 参数、梯度和 Adam；允许 TF32，无 BF16 autocast |

所有 Adam／反向次数按不触发 KL 提前停止计算。保留 target KL=0.05 和梯度裁剪 1.0。
完整本轮缓存约 0.817 GiB CPU RAM，不含环境、索引、解包和进程本身；只保留当前 rollout 并复用。

SB3 在采样后更新学习率，因此初始配置为 3e-4，首次 PPO 训练使用 2.9703e-4，
第 50 次为 1.515e-4，第 100 次为 3e-6。同一次 PPO 中的 32 次 Adam 使用同一个 LR。

## 循环层级

```python
for ppo_iteration in range(100):
    rollout = collect(n_envs=256, n_steps=1024)  # 262,144 条新样本
    for epoch in range(1):
        shuffle(rollout)
        for minibatch in split(rollout, 8192):  # 32 个
            optimizer.zero_grad()
            for microbatch in split(minibatch, 1024):  # 8 个
                normalized_loss(microbatch).backward()
            clip_gradients()
            optimizer.step()
```

伪代码省略 KL 早停和其他损失细节；归一化按完整有效 minibatch 计算，不能直接把辅助损失机械地除以 8。
epoch 在 PPO iteration 内、minibatch 循环外；microbatch 循环结束后才执行 Adam。
epoch=2 会复用同一批新数据两次，而不是再次采样，并将名义 Adam 次数翻倍。

## epoch 与采样的取舍

以相同有效 minibatch=8,192、相同每次 PPO 的 32 次 Adam 更新为例：

| 方案 | 每次 PPO 新样本 | minibatch／epoch | epochs | Adam／PPO |
| --- | ---: | ---: | ---: | ---: |
| 当前选择：更多新数据 | 262,144 | 32 | 1 | 32 |
| 对照：复用较少新数据 | 131,072 | 16 | 2 | 32 |

epoch=1 可以作为当前起点，但两者并不等价：前者需要两倍新采样，后者在相同数据上重复优化。
固定 minibatch 大小而增加数量，也会扩大 rollout，使采样策略更久不刷新，并增加缓存需求。
所以不把“采样便宜”直接解释为 minibatch 越多越好，也不保证 epoch=1 一定更优。

上一版 100-minibatch 试跑实测累计采样约 132 秒、训练更新约 73 秒；
采样包括策略前向、环境推进、传输和缓存写入，不能只用原生模拟器吞吐推断采样便宜。
旧大 rollout 的完整迭代则约为采样 249 秒、更新 370 秒，硬件和批量设置都会改变瓶颈。
应在该配置实际运行后重新观察采样／更新占比、KL、裁剪比例、value 拟合和固定评估名次。

## 起点、评估与留存

本次从 `runs/20261031-ppo-small-init/initial_model.zip` 的零步初始化开始。
32 个 mini 不是上一版总共 100 个 mini 的预算；本次总共有 3,200 个有效 mini 的处理量。
模型位于 `runs/20261031-ppo-small-100ppo-32minibatches/`。

沿用固定评估协议：初始 64 局，每 25 次 PPO 迭代评估 64 局并保存定期检查点，
结束后用另一组种子评估 128 局。按固定评估集选择 best，独立种子用于复核，不再次用于选模。
按用户留存要求，评估完成后只保留本实验 best_model.zip；指标、W&B 记录、历史原始日志和清理摘要保留。
训练入口会先写临时的 starting/latest/best/final/定期模型，留存清理是评估后的独立步骤。

配置和整除核验见 [config.json](config.json)、[validation.json](validation.json)。
论文与其他 PPO 项目的单位对照见 [统一对照表](../../../docs/PPO_BATCH_COMPARISON.md)。

## 启动命令

监督进程 run.py 已完成训练和 finalize.py，核验实际 Adam 次数、独立评估和最佳模型，并清理了其他本实验模型。
异常或中断时不会清理模型。不要重复执行已有模型目录的启动命令。

本次监督入口：

```bash
uv run --locked --group train --extra test \
  python experiments/20261031-exp01-ppo-scripted/20261031-ppo-small-100ppo-32minibatches/run.py
```

底层训练入口如下：

```bash
uv run --locked --group train --extra test python -m taverngym.learning.experiment \
  --config experiments/20261031-exp01-ppo-scripted/20261031-ppo-small-100ppo-32minibatches/config.json \
  --model runs/20261031-ppo-small-init/initial_model.zip \
  --experiment-dir experiments/20261031-exp01-ppo-scripted/20261031-ppo-small-100ppo-32minibatches \
  --run-dir runs/20261031-ppo-small-100ppo-32minibatches --device cuda
```

## W&B 记录粒度与迁移

按用户要求改用 [W&B 在线 run](https://forge.coreweave.com/wandb/eric_eric_/AutoTavern/runs/4xw0tzgd)，
并已执行 `weave.init('eric_eric_/AutoTavern')`。TensorBoard 界面服务已关闭。
训练进程保持运行，通过旁路进程补录并持续同步已有日志；模型与 Adam 未重启。
原始 config.json 中的 TensorBoard 字段保留启动时真实配置，迁移单独记录在 tracking-migration.json。
后续新训练入口原生写入 W&B。

**横轴统一使用实际累计 Adam optimizer step**，不沿用旧试跑的“累计采样量”横轴。

| 曲线 | 记录时机 | 预计点数 |
| --- | --- | ---: |
| `train/*` | 每次实际执行 Adam 后，记录该 minibatch 的更新前 loss、KL、熵、梯度范数与 LR | 最多 3,200 |
| `train/mean_placement` 等 | 每次 PPO 采样中已结束游戏的结果 | 100 |
| `train/ppo_*` | 每次 PPO 迭代结束，记录本轮汇总和实际 Adam 计数 | 100 |
| `val/*` | 训练前 step=0，加 PPO 第 25、50、75、100 次后的固定 64 局评估 | **5** |
| `test/*` | 完成后对 final、预选 best、初始化在独立 128 局种子上复核 | 每个独立版本／指标 1 |
| `system/*` | 每 PPO 采样／更新耗时；SDK 另采硬件状态 | 每阶段耗时 100 |

8 次 microbatch 只累积梯度，不分别记录为 Adam 点；触发 KL 早停而未执行 Adam 的 minibatch 也不计为 train 点。
实际 train 点数可能少于 3,200。`train/ppo_iteration` 和 `train/sampled_decisions` 提供对应 PPO 迭代及累计新样本数。
每个 PPO 的实际 Adam 次数与累计数也写入 metrics.jsonl 和 progress.json。

baseline 已合入同一条 val 曲线的 step=0，后续仍为四次验证；所有指标只用一层 `/`。
独立 test 不参与选模，也不混入 val。
原始 TensorBoard 文件和旧试跑指标保留；旁路将 `baseline/* → val/*`、`rollout/* → train/*`，
`ppo_summary/* → train/ppo_*`、`time/* → system/*`。
指标逐项含义、必要性、恢复同步命令和 CPU/GPU 快照见 [训练指标文档](../../../docs/TRAINING_METRICS.md)。

论文图 2 为每个发布版本的吃鸡率（每点 24,576 局），图 3 为真实对局后的天梯分数，图 6 为游戏内回合的阵容强度。
论文没有公开“每次 Adam 一个 loss 点”的 TensorBoard 记录规则；以上是本地日志设计，非论文复现声明。
来源：[论文图 2、3、6](https://arxiv.org/html/2610.33845v1)。

当前模型在训练后按命名约定改为 `20261031-131146-best_model.zip`，前缀取训练开始的北京时间；文件内容与 SHA-256 保持不变。重命名记录见 artifact-renaming.json。
