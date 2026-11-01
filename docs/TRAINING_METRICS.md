# W&B 与训练指标

2026-10-31 按用户要求切换到 W&B，项目为 `eric_eric_/AutoTavern`。
配置见 [tracking.json](../configs/tracking.json)，精确 SDK 版本由 uv.lock 锁定。
W&B 负责 PPO 标量曲线，Weave 使用同一个项目初始化；仅初始化 Weave 不会自动记录 PPO 指标。

```python
wandb.init(entity="eric_eric_", project="AutoTavern", ...)
weave.init('eric_eric_/AutoTavern')
```

Weave 初始化遇到网络／TLS 错误时，错误写入 `wandb.json` 的 `weave_error`，
`weave_initialized=false`；W&B 的 PPO 标量日志继续正常写入。
调用仍使用上述项目名，失败状态不会伪报为已连接，也不会阻断训练采样。

新训练的 `learning.experiment` 和 `learning.train` 默认直接调用 `wandb.log`，
同时写本地 `experiments/<name>/wandb-metrics.jsonl`，不生成 TensorBoard 事件文件。
初始化模型本身不启动在线 run。模型仍保存在 `runs/<name>/`。
`wandb.json` 保存 run ID 和链接；续训复用同一个 run。
鉴权沿用机器已有 W&B 凭据，不写入仓库或实验配置。

## 关键测评指标与 best 选择

V7 移动成本准备版（2026-11-01，未启动）新增独立 `move_cost=0.0005`，不复用同时处罚
Move／Freeze 的 `operation_cost`。保留循环成本 0.01；实际移动又命中循环时合计扣 0.0105。
新增 `move_actions`（每局实际招募 Move 次数）、`move_penalty`（累计移动罚分）、
`move_action_fraction`（先算每局 Move／策略转移数，再按局平均），进入 train／val／greedy。
`total_reward = placement_reward + operation_penalty + cycle_penalty + move_penalty`。
历史缺 `move_cost` 等同 0，不补造旧日志的移动指标；这些新增观察不改变随机平均名次严格降低的 best 规则。
准备方案见 [V7 配置说明](../experiments/20261101-exp07-v7-history-ranked/20261101-ppo-v7-move0005-cycle001-forceend/README.md)。

当前 V3 的混合验证对手为 **2 个固定 V1 + 3 个固定 V2 + 2 个脚本**，
学习者随机／贪心各 **64 局／验证点**，所有模型 BF16 前向；固定模型对手始终随机采样。
后续实验须按自己的对手与局数解释指标，不能照搬这个阵容。

| 指标 | 如何理解 | 用途／是否参与选 best |
| --- | --- | --- |
| `val/mean_placement` ↓ | 随机采样组所有验证局的最终名次平均值 | **唯一选模指标，越低越好** |
| `val/first_rate` ↑ | 第一名局数／验证局数 | 观察吃鸡能力；不参与选模，也不用于打破平局 |
| `val/top4_rate` ↑ | 前四名局数／验证局数 | 观察稳定性；不参与选模 |
| `val/greedy_mean_placement` ↓ | 同一混合对手下，学习者每次选概率最高合法动作时的平均名次 | 观察贪心推理效果；当前不参与选模 |
| `val/cycle_returns` ↓、`val/greedy_cycle_returns` ↓ | 每局被奖励检测器识别的可见状态返回次数，再对验证局求平均 | 行为质量诊断；受 `cycle_detection` 模式影响，不参与选模 |
| `val/cap_end_rate` ↓、`val/greedy_cap_end_rate` ↓ | 每局因操作预算强制 End 的回合比例，再对验证局求平均 | 检查对外部上限的依赖；不参与选模，不等于胜率 |
| `val/actions_per_game` | 每个学习者 episode 的操作数均值 | 解释操作成本与吞吐；没有越少越好的绝对目标 |
| `val/total_reward` ↑ | 名次效用加操作／循环惩罚后的回报 | 解释训练目标；当前不用它选 best |
| `train/loss`、`train/approx_kl`、`train/learning_rate` | 优化损失、策略漂移、实际学习率 | 训练健康检查，不能替代对战实力评估 |
| `test/*` | 独立种子上的 initial／best／final 结果；V3 计划各条件各 128 局 | 最终复核，不反过来选模 |

`mean_placement = 最终名次之和 / 对局数`，取值在 1 到 8 之间。
例如四局第 1、2、4、5 名，平均名次为 `(1+2+4+5)/4 = 3.0`。
它不是胜率，2.875 也不表示每局都拿第三名。

代码仅在新验证的 `mean_placement` **严格小于**此前最佳值时覆盖 best；
相等保留较早版本，不根据 first/top4、贪心表现或循环次数追加排序。
step=0 的继承模型也参加比较，因此续训没有进步时可保留起点。
贪心验证同样不构建梯度，区别是选动作方式；最佳模型当前依据随机采样组。

未来改变选模目标（例如改为 Greedy 或加入循环约束）必须另写实验配置与规则，
不能修改既有 run 的历史结论。64 局估计有波动，应连同对手、模式、局数和种子报告。
实现见 [evaluate_suite 与 Recorder](../python/taverngym/learning/experiment.py)。

V4–V6 显式启用三次完整循环与第 61 次回合动作请求判负后，新增
`cycle_forfeit_rate`、`action_limit_forfeit_rate`，分别表示完成局中两种判负的比例。
对应 `*_forfeit_turn`、`*_forfeit_placement` 对未触发局填零；其全局均值除以对应 rate，
才得到触发局的条件平均回合／名次。`max_consecutive_cycles` 是每局连续闭环数的峰值。
这些键进入 `train/*`、随机 `val/*` 和 `val/greedy_*`，仅作观察，不改变选 best 规则。
`cap_end_rate` 继续只统计实际强制 End，超限请求被拦截判负不计入此指标。
原始终局 info 记录 `termination_reason`，区分 `cycle_forfeit` 与 `action_limit_forfeit`。

此处“循环”指可见状态闭环，`cycle_returns` 仍是每次返回已见状态的次数，两者不能混用。
V3 起点须在 V4 新规则下重评后比较；原 V3 历史结果不回写。

## 横轴与分组

所有曲线的默认横轴是 **实际累计 Adam optimizer step**，键为 `optimizer_step`。
每个有效 minibatch 执行一次 Adam 后记录一个 train 点；microbatch 只累积梯度。
KL 提前停止而未执行的 Adam 不计点。记录的 loss、KL 和梯度对应这次更新前的 minibatch。
另存累计采样量和 PPO iteration，便于换算；W&B 内部 `_step` 是日志行号，不是 Adam 计数。

标签只保留一层 `/`：`train/*`、`val/*`、`test/*`、`system/*`。
rollout 的对局结果放入 `train/*`；baseline 合入 `val/*` 的 step=0。
`val/mean_placement` 因此从起点连续显示到后续评估，没有单独 baseline 曲线。

### 汇报时明确对手

之后每次汇报、结果表和新实验图表标题都写明实际对手组成、固定模型版本及学习者的选动作方式，
例如“2 个固定 V1 + 5 个脚本，学习者随机采样，64 局”。不能只用 `val`、`mixed` 或“胜率”代替评测条件。
跨时间比较须确认对手组成与策略模式一致；baseline 另外注明学习者权重来源，step=0 仅指当前阶段尚未更新。

第二轮 2 V1 实验的历史对应关系如下；其他 run 必须按各自配置说明，不能照搬对手数量：

| 当前指标前缀／示例 | 对手 | 学习者选动作 |
| --- | --- | --- |
| `val/mean_placement`、`val/first_rate`、`val/top4_rate` | 2 个固定 V1 + 5 个脚本 | 随机采样 |
| `val/greedy_*` | 2 个固定 V1 + 5 个脚本 | 贪心 |
| `val/scripted_*`，不含下一行 | 7 个脚本 | 随机采样 |
| `val/scripted_greedy_*` | 7 个脚本 | 贪心 |

同组其他指标沿用相同对手条件。混合组中的固定 V1 对手均随机采样，`greedy` 只改变学习者。
后续新实验的图例也应直接标注对手组合；当前运行中的指标键继续沿用已有名称，以保持曲线连续。

后续新混合实验设置 `evaluation_include_scripted=false`，只保留前两类混合对手曲线：
`val/*` 为学习者随机采样，`val/greedy_*` 为学习者贪心，两组对手组成相同。
不再追加七脚本随机／贪心对照，不影响混合阵容本身的脚本数量。
各曲线仍包含 step=0 和原定周期验证点；缺少开关的历史配置保留原四组协议。

第一轮历史 100 PPO 实验采用 256 环境 × 1,024 步、每 PPO 最多 32 次 Adam：

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
各实验使用本地配置的脚本或冻结模型对手，分数不是论文校准对手库上的结果。

## 启用操作／循环成本后的指标

以下指标仅在配置 `reward_shaping` 后记录。`train/*` 是本轮采样期间**已完成整局**的均值，
每 PPO 一点，包含这局跨越此前 rollout 的操作；`val/*` 是对应固定验证局的均值，含 step=0。
这些是每局累计值的平均，而非每个动作的均值，不与每 Adam 的 loss 混合。

| `train/`、`val/` 下的同名指标 | 含义 |
| --- | --- |
| `placement_reward` | 原始终局名次效用，始终与真实名次对应 |
| `operation_penalty` | 每局 Move / Freeze 的累计负成本；后续仅循环配置设 `operation_cost=0`，此项为 0，历史轮次保持旧值 |
| `cycle_penalty` | 每局无进展状态往返的累计负成本 |
| `total_reward` | 上述三项之和；训练的 GAE / critic 使用这个奖励 |
| `reversible_actions` | 每局 Move / Freeze 总次数 |
| `cycle_returns` | 每局返回已见完整可见状态的次数；连续返回可以重叠计数；覆盖范围由 `cycle_detection` 指定 |
| `actions_per_game` | 每局全部决策操作数 |
| `cap_end_rate` | 每局达到动作上限后被迫 EndTurn 的招募回合比例，再取整局均值 |

第二轮增加四个验证条件，主组沿用 `val/*`；混合贪心、全脚本随机、全脚本贪心
分别使用 `val/greedy_*`、`val/scripted_*`、`val/scripted_greedy_*`，各有初始点加 4 次周期点。
训练采样默认随机，冻结旧模型只推理，`train/opponent_decisions` 记录其每轮操作数，
不把它们计入学习者 sample 或 Adam step。

本地 `metrics.jsonl` 中对应键为 `rollout/*`，W&B 会映射为一层 `/` 的 `train/*`。
验证环境只计算这些指标，不更新模型或过滤动作；选 best 仍依据 `val/mean_placement`。
相同种子、相同动作下，启用奖励不会改变模拟器状态、合法掩码、终局名次或验证随机流。
这项改动不会补写第一次实验的历史 W&B 曲线；下一轮启动时会直接记录。

## 第一轮历史日志迁移

历史实验：[100 PPO × 32 minibatches](../experiments/20261031-exp01-ppo-scripted/20261031-ppo-small-100ppo-32minibatches/README.md)。
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
  --experiment-dir experiments/20261031-exp01-ppo-scripted/20261031-ppo-small-100ppo-32minibatches
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
