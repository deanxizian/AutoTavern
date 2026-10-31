# W&B 与训练指标

2026-10-31 按用户要求切换到 W&B，项目为 `eric_eric_/AutoTavern`。
配置见 [tracking.json](../configs/tracking.json)，精确 SDK 版本由 uv.lock 锁定。
W&B 负责 PPO 标量曲线，Weave 使用同一个项目初始化；仅初始化 Weave 不会自动记录 PPO 指标。

```python
wandb.init(entity="eric_eric_", project="AutoTavern", ...)
weave.init('eric_eric_/AutoTavern')
```

新训练的 `learning.experiment` 和 `learning.train` 默认直接调用 `wandb.log`，
同时写本地 `experiments/<name>/wandb-metrics.jsonl`，不生成 TensorBoard 事件文件。
初始化模型本身不启动在线 run。模型仍保存在 `runs/<name>/`。
`wandb.json` 保存 run ID 和链接；续训复用同一个 run。
鉴权沿用机器已有 W&B 凭据，不写入仓库或实验配置。

## 横轴与分组

所有曲线的默认横轴是 **实际累计 Adam optimizer step**，键为 `optimizer_step`。
每个有效 minibatch 执行一次 Adam 后记录一个 train 点；microbatch 只累积梯度。
KL 提前停止而未执行的 Adam 不计点。记录的 loss、KL 和梯度对应这次更新前的 minibatch。
另存累计采样量和 PPO iteration，便于换算；W&B 内部 `_step` 是日志行号，不是 Adam 计数。

标签只保留一层 `/`：`train/*`、`val/*`、`test/*`、`system/*`。
rollout 的对局结果放入 `train/*`；baseline 合入 `val/*` 的 step=0。
`val/mean_placement` 因此从起点连续显示到后续评估，没有单独 baseline 曲线。

当前 100 PPO 实验采用 256 环境 × 1,024 步、每 PPO 最多 32 次 Adam：

| 组 | 记录时机 | 点数／横轴 |
| --- | --- | --- |
| `train/loss` 等 minibatch 指标 | 每次实际 Adam | 最多 3,200 点，step=1…3,200 |
| `train/mean_placement` 等采样对局结果 | 每个 PPO 结束 | 100 点，位于该 PPO 最后一个 Adam step |
| `train/ppo_*` 汇总 | 每个 PPO 结束 | 100 点；避免与每 Adam loss 混成同一条线 |
| `val/mean_placement`、`val/top4_rate`、`val/first_rate` | 初始 64 局 + PPO 25/50/75/100 各 64 局 | **5 点**，无 KL 早停时为 0/800/1,600/2,400/3,200 |
| `test/{initial,best,final}_*` | 完成后独立 128 局复核 | 每个版本每指标 1 点，不参与选模 |
| `system/*` | 每个 PPO 结束；SDK 另采硬件状态 | 采样与更新耗时各 100 点 |

## 指标含义与必要性

| 指标 | 含义与用途 | 必要性 |
| --- | --- | --- |
| `val/mean_placement` ↓ | 固定评估种子下平均名次；选择 best 的主指标 | 必需 |
| `val/top4_rate`、`val/first_rate` ↑ | 稳定前四与争夺第一的能力；防止只看平均名次 | 必需 |
| `val/ppo_iteration` | 这个评估点对应的 PPO 轮数 | 定位辅助 |
| `train/mean_placement`、`train/top4_rate`、`train/first_rate` | 本轮采样期间已结束对局的结果；策略和局面分布随训练变化 | 有用，不能替代固定 val |
| `train/loss` | policy + 0.5×value + 0.002×entropy + 0.1×placement 的总损失 | 必需，检查训练异常；不是实力评分 |
| `train/policy_gradient_loss` | PPO 裁剪策略目标的损失；受优势归一化影响 | 有用 |
| `train/value_loss` | 价值预测与 GAE returns 的均方误差 | 必需，检查 critic 拟合 |
| `train/entropy_loss` | 负策略熵；接近 0 常表示策略更集中，也受合法动作数影响 | 必需，监控探索 |
| `train/approx_kl` | 当前 minibatch 在更新前相对 rollout 策略的近似 KL | 必需，监控策略漂移与早停 |
| `train/clip_fraction` | 概率比超出 1±0.2 的样本占比 | 必需，判断裁剪是否频繁 |
| `train/grad_norm` | 裁剪前总梯度范数；随后按上限 1.0 裁剪 | 必需，检查梯度稳定性 |
| `train/learning_rate` | 该次 Adam 实际使用的学习率 | 必需，核验线性 scheduler |
| `train/placement_loss` | 已知终局名次片段的辅助分类损失 | 使用辅助头时必需 |
| `train/placement_label_fraction` | minibatch 中已知终局名次标签的样本比例 | 有用，解释辅助损失的覆盖率 |
| `train/ppo_explained_variance` | rollout 时价值预测解释 returns 方差的比例；不是更新后重新预测 | 有用，检查 critic |
| `train/ppo_*loss`、`train/ppo_approx_kl`、`train/ppo_clip_fraction` 等 | 每轮汇总；包含参与 loss 计算但可能因 KL 被拒绝更新的最后一个 minibatch | 有用，看整体趋势 |
| `train/ppo_optimizer_steps`、`train/ppo_optimizer_steps_this_iteration` | 实际累计 Adam 数及本轮 Adam 数 | 必需，核对预算与 KL 早停 |
| `train/ppo_iteration`、`train/sampled_decisions` | 当前 PPO 轮数与累计新样本量 | 必需，换算进度 |
| `train/ppo_clip_range` | PPO 裁剪系数，目前固定为 0.2 | 配置审计 |
| `train/ppo_learning_rate`、`train/ppo_placement_label_fraction` | 同一数据的每 PPO 汇总版本 | 趋势辅助 |
| `system/rollout_seconds`、`system/update_seconds` | 采样和训练更新的墙钟耗时 | 必需，定位吞吐瓶颈 |
| `test/*` | 未用于选模的独立种子结果，分开报告 initial/best/final | 必需，复核泛化 |

固定验证集用于选模，test 不参与选择。64 局验证的估计有波动，应连同局数和种子解读。
当前规则与对手为本地脚本基线，分数不是论文校准对手库上的结果。

## 当前训练的迁移

当前实验：[100 PPO × 32 minibatches](../experiments/20261031-ppo-small-100ppo-32minibatches/README.md)。
[在线 run](https://forge.coreweave.com/wandb/eric_eric_/AutoTavern/runs/4xw0tzgd)。

这次训练启动时已在写两个 TensorBoard 文件。保留训练进程，使用独立进程读取旧文件，
补录历史并持续同步 W&B；TensorBoard UI 服务已关闭。
原始启动配置保持原值，迁移另外记录在实验目录的 tracking-migration.json。
两个仍在写入的文件分别读取，避免目录读取器漏掉较早文件的新事件。
重启同步进程时按本地 journal 的“规范化指标名 + Adam step”去重。

映射为 `baseline/* → val/*`、`rollout/* → train/*`、
`ppo_summary/* → train/ppo_*`、`time/* → system/*`。
完成后的旧版 finalizer 仍会写原始 test 事件，旁路持续到评估和留存步骤结束后再退出。
模型、采样状态、随机流和 Adam 均未因为日志迁移重启或重新加载。

迁移进程中断时可从已有 run ID 与 journal 恢复，先确认旧同步进程已退出：

```bash
uv run --locked --group train python -m taverngym.learning.migrate_wandb \
  --experiment-dir experiments/20261031-ppo-small-100ppo-32minibatches
```

不对已运行的训练热替换 Python 代码。未来新训练直接写 W&B；
`tracking.mode=offline` 可离线保存，兼容测试显式使用 `tracking_backend=tensorboard`。
SDK 缓存 `wandb/`、`.weave/` 被 Git 忽略，指标 journal 和配置记录保留。

本次验证为 **62 个 Python 测试通过**，覆盖双事件文件持续更新、同步去重、
原生 W&B 离线记录的每 Adam 点数、step=0 与周期验证合并，以及保存／加载后不携带日志客户端。
W&B 离线 SDK 仍需本地 socket；测试不向在线项目发送数据。
`uv lock --check --offline` 和 `git diff --check` 通过。

## CPU/GPU 快照

本实验 2026-10-31 13:24 的 16.2 秒、15 次观测，原始记录在实验目录
profiling/system-utilization-20261031T132458.json。它是当时的短窗口测量。

| 项目 | 测量结果 |
| --- | ---: |
| 机器整体 CPU（8 个逻辑核） | 平均 20.4% |
| 训练进程 CPU | 平均 132.7%，即约 1.33 个逻辑核 |
| 训练进程 RSS | 3.28 GiB |
| 系统已用／总内存 | 4.88 / 47.05 GiB |
| GPU 利用率（采样窗口） | 平均 40.6% |
| GPU 利用率（更新窗口） | 平均 89.0% |
| GPU 利用率（全窗口） | 平均 59.9%，峰值 96% |
| GPU 显存已用／总量 | 7.42 / 15.92 GiB |
| GPU 平均功耗 | 185 W |

采样与更新当前串行交替。GPU 在更新阶段更忙，采样时较空闲；
CPU 内存和显存均有余量。该窗口的利用率不能直接当作 FLOPS 利用率，
也不足以证明增加 CPU 线程或 batch 一定提速，阶段耗时仍应看 system 曲线。
