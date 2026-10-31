# PPO 开发环境与训练产物

本轮改进、当前默认值、实测边界与 Git 忽略策略见 [训练基础设施](TRAINING_INFRA.md)。
PPO iteration、epoch、sample、minibatch 和 Adam step 的区别，以及论文／其他项目的统一对照，
见 [PPO 批次与预算对照表](PPO_BATCH_COMPARISON.md)。
指标含义、记录频率及当前训练的在线迁移见 [W&B 与训练指标](TRAINING_METRICS.md)。
现有模型的动作往返问题及可选推理筛选，对照见
[循环诊断](../experiments/20261031-exp01-ppo-scripted/20261031-action-loop-diagnostic/README.md) 和
[筛选实验](../experiments/20261031-exp01-ppo-scripted/20261031-action-loop-filter/README.md)。
该选项需要调用方显式接入，未改默认策略或模型权重。

下一轮已确认使用训练奖励减少无效操作，见
[实验 02 奖励配置](../experiments/20261031-exp02-mixed-opponents/README.md) 和
[实现验证](../experiments/20261031-infra/20261031-reward-shaping/README.md)。
新初始化模板 `configs/ppo-small.json` 启用操作成本 `0.0001` 与循环成本 `0.0002`；
旧实验配置与既有模型保持历史奖励，缺少 `reward_shaping` 或设为 `null` 时不启用。

正式实验入口 `learning.experiment` 读取独立实验配置中的这一段：

```json
"reward_shaping": {"operation_cost": 0.0001, "cycle_cost": 0.0002}
```

它不会从旧模型中自动推断这一设置；下一轮完整配置需合入已确认的配置片段。
通用入口 `learning.train` 沿用检查点配置，也可用 `--operation-cost 0.0001 --cycle-cost 0.0002`
为旧检查点显式开启；两个参数均为非负的成本绝对值，实际 reward 中相减。
所有设置写入实验配置与新检查点的 `experiment_config`。
奖励在环境采样时加入，不更改动作或掩码，推理无须接入额外筛选器。

## 前向精度

2026-10-31 后续配置确认采用 BF16 autocast 前向，FP32 参数、累积梯度、Adam、
head 输出、概率／log probability、GAE、returns 和 loss。
BF16 的反向算子随前向算子精度执行，最终累积到 FP32 参数梯度；不启用 GradScaler。
动作索引、卡牌 ID 和布尔掩码保持其原有整数／布尔类型，lossless rollout 编码不变。

```json
"forward_precision": "bf16",
"opponent_forward_precision": "bf16"
```

学习者采样、训练重算、预测和 bootstrap value 统一通过 policy 内的前向入口。
固定对手的精度缺省跟随学习者；显式 `opponent_forward_precision` 可单独覆盖。
`precision_settings` 会记录到 resolved-config、W&B 配置和新检查点的实验配置，
模型的 `policy_kwargs.forward_precision` 会随 SB3 文件保存／加载。

新初始化模板启用 BF16；旧配置和旧检查点未设置时仍沿用 FP32。
`learning.experiment` 和 `learning.train` 支持 `--forward-precision bf16` 显式覆盖旧模型，
未指定则继承配置／检查点。直接使用 Python 时可调用 `model.set_forward_precision("bf16")`；
`load_frozen_policy(path, device, forward_precision="bf16")` 可覆盖旧对手文件的计算精度。
这些调用不会把权重转换成 BF16，也不会改写源检查点。

BF16 前向概率差异的整局对照见
[第一轮精度评估](../experiments/20261031-exp01-ppo-scripted/20261031-bf16-match-evaluation/README.md)。
第二轮已启动 5 脚本 + 2 V1 + 1 学习者，使用 BF16 与独立 Actor 调度，配置入口见
[next-settings.json](../experiments/20261031-exp02-mixed-opponents/next-settings.json)。
原有 4+3 试跑的配置与停训检查点不自动转换、不自动续训。

后续训练默认使用 **W&B**，项目 `eric_eric_/AutoTavern`，并调用
`weave.init('eric_eric_/AutoTavern')`。不再启动 TensorBoard 或为新训练生成事件文件。
最近完成的一次训练通过旁路进程补录、持续同步原有日志，训练进程未重启。

最近一版已完成 **100 次 PPO iteration、每个 epoch 32 个 minibatch、epoch=1**，
共 26,214,400 条新样本、3,200 次 Adam 更新，总耗时约 57 分钟。
最终独立 128 局平均名次 1.765625、前四率 93.75%、第一名率 58.59375%，对手为七个脚本机器人。
参数、循环层级和结果见 [实验记录](../experiments/20261031-exp01-ppo-scripted/20261031-ppo-small-100ppo-32minibatches/README.md)。

已完成的首版采用用户指定的小预算：100 个有效 minibatch，共 100 次 Adam 更新、819,200 条样本，
microbatch=1,024、累积 8 次，学习率配置 `3e-4 → 3e-6` 线性衰减，无 warmup。
从既有零步初始化开始，采用 32 环境 × 256 步，每轮一个有效 minibatch；
配置、命令及结果见 [首版实验](../experiments/20261031-infra/20261031-ppo-small-v1-100-minibatches/README.md)。
下文的旧 300 轮实验保持停止，其配置与实测结果保留历史值。

## 安装

使用 uv 管理项目独立的 `.venv`，Python 版本由根目录的 `.python-version` 指定为 3.12。
训练依赖在 `pyproject.toml` 的 `train` 组中，精确版本由 `uv.lock` 锁定。

```bash
uv sync --locked --python 3.12 --group train --extra test
uv run --locked --group train --extra test python --version
```

从源码安装 TavernGym 需要 `rust-toolchain.toml` 指定的 Rust 工具链。
本次安装使用机器已有的 Python 3.12.8，未修改系统或 Conda 基础环境。
当前会话的 Rust 位于 `/tmp/autotavern-cargo` 和 `/tmp/autotavern-rustup`；如需在
同一会话中重新编译绑定，可使用：

```bash
PATH=/tmp/autotavern-cargo/bin:$PATH \
RUSTUP_HOME=/tmp/autotavern-rustup \
CARGO_HOME=/tmp/autotavern-cargo \
RUSTUP_TOOLCHAIN=1.98.1 \
UV_CACHE_DIR=/tmp/autotavern-uv-cache \
uv sync --locked --python 3.12 --group train --extra test
```

`/tmp` 工具链和缓存是当前机器的临时安装位置；其他机器应先安装 Rust 工具链。

训练依赖包括 PyTorch、Stable-Baselines3、sb3-contrib（提供 MaskablePPO）、
Gymnasium、W&B 和 Weave；TensorBoard 仅保留作旧日志读取和兼容验证。
运行训练相关命令时保留 `--group train`；
该组不是默认依赖组，普通 `uv sync` 可能移除训练依赖。

安装验证及实测版本见
[20261031-ppo-library-setup](../experiments/20261031-infra/20261031-ppo-library-setup/README.md)。

## 模型与日志位置

后续训练按实验开始日期（Asia/Shanghai）和实验名建立目录：

```text
runs/yyyymmdd-{exp-name}/
├── checkpoints/       # 定期保存的检查点
├── YYYYMMDD-HHMMSS-best_model.zip # 按评估指标选出的最佳模型，北京时间
├── latest_model.zip   # 最近一次完整 PPO 更新后的模型
└── final_model.zip    # 训练结束时的模型

experiments/yyyymmdd-{实验名称}/yyyymmdd-{记录名称}/
├── README.md          # 目的、方法、命令、结论及模型路径
├── experiment.json    # 实验状态与产物索引
├── config.json        # 超参数、随机种子、环境与对手设置
├── progress.json      # 当前进度
├── metrics.jsonl      # 每轮指标
├── wandb.json         # W&B run ID、项目与链接，恢复时复用
├── wandb-metrics.jsonl # 已提交的标量指标，含实际 optimizer_step
├── wandb/             # SDK 缓存与同步状态，本地忽略
└── ...                # 评估指标与原始文本输出
```

已保存第一版零步初始化模型：
`runs/20261031-ppo-small-init/initial_model.zip`，该文件保持零训练步数。
实验配置、W&B 日志和指标全部放在 `experiments/`，`runs/` 存放模型。
基础设施和试跑归入 `experiments/20261031-infra/`，独立脚本基线归入 `experiments/20261031-scripted-baseline/`；
第一次正式训练与其 benchmark 集中在 `experiments/20261031-exp01-ppo-scripted/`，
第二次预留为 `experiments/20261031-exp02-mixed-opponents/`。
导航与目录约定见 [实验索引](../experiments/README.md)。
历史 TensorBoard 文件保留作为来源记录。
后续最佳模型文件名以训练开始的北京时间（精确到秒）为前缀，例如
`20261031-150000-best_model.zip`。同次训练刷新 best 时覆盖同名文件，
续训沿用检查点中记录的文件名；实际名称写入 `resolved-config.json` 的
`best_model_filename` 和进度记录的 `best_model`，供评估及最终留存读取。
旧检查点兼容原来的 `best_model.zip` 名称。
`runs/` 已由 Git 忽略；`experiments/` 保存可审查的记录。
配置还应记录代码版本、依赖锁文件摘要、观测编码、动作掩码与奖励定义，
以便正确加载模型并比较实验。

`.zip` 是 SB3 的原生保存格式，`model.save(path)` 会自动生成，包含网络参数、
优化器状态和算法配置；使用 `MaskablePPO.load(path)` 加载。
检查点不自动包含 TavernGym 当前游戏、所有随机数状态或外部归一化器状态；
若实验要求精确续跑，需要另行保存这些状态。

## 第一版小模型

当前使用 [configs/ppo-small.json](../configs/ppo-small.json)：`d=192`、4 层、
8 个注意力头、FFN 宽度 1024，实际 **3,491,978 个可训练参数**。
完整说明、实现选择与验证结果见
[20261031-ppo-small-init](../experiments/20261031-infra/20261031-ppo-small-init/README.md)。

网络实现位于 `python/taverngym/learning/`：身份 embedding、静态特征投影、
冻结的规则文本向量投影、实体 Transformer、结构化动作评分、价值和名次头。
Gymnasium 包装提供空间与 `action_masks()`，`EntityPPO` 在 MaskablePPO 上
增加名次辅助损失，仅对本轮采样中已知终局名次的片段监督。

规则文本使用 [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
生成 384 维英文规则向量，模型版本已固定；离线文本编码器不计入在线决策网络参数量。
文本向量已包含在 `.zip` 内，加载训练不需要重新下载编码器或生成缓存。
MiniLM、FFN、汇聚和指针打分公式等属于我们补充的实现选择，论文未完整指定。

基础设施的大 rollout 预演参考论文早期规模对比预算：原为 400 轮，后改为 300 轮，每轮 4,194,304 个决策。
配置、实时状态、无损采样压缩及梯度累积说明见
[20261031-ppo-small-paper400](../experiments/20261031-infra/20261031-ppo-small-paper300/README.md)。
当前训练已按用户要求停止，保留 3 轮检查点。以下为原始启动命令，仅供参考，不会自动执行：

```bash
uv run --locked --group train --extra test python -m taverngym.learning.experiment \
  --config experiments/20261031-infra/20261031-ppo-small-paper300/config.json \
  --model runs/20261031-ppo-small-init/initial_model.zip \
  --experiment-dir experiments/20261031-infra/20261031-ppo-small-paper300 \
  --run-dir runs/20261031-ppo-small-paper400 --device cuda
```

该入口保留初始化网络权重，覆盖采样、batch、学习率等实验设置：256 个环境、
每环境 16,384 步；有效 batch 8,192，默认 microbatch 1,024 × 梯度累积 8；
学习率参考表 7 的端点，线性 `6e-6 → 5e-6`。
模型保存至 `runs/`，配置、TensorBoard、控制台、每轮进度与评估保存至 `experiments/`。
七个对手为相同的脚本策略，没有前一阶段的 KL 锚定。
论文早期完整配方未公开，这仍是与论文预算对齐的本地脚本对手基线。

2026-10-31 将默认 microbatch 从 256 改为 1,024；已有三轮检查点、运行时配置和性能原始记录保留原值。
`EntityPPO` 新建模型、通用训练入口以及正式实验配置采用新默认，加载旧检查点时由训练入口覆盖旧值。
初始化模板的有效 batch 为 256，实际切片最多为该 batch 大小；正式实验的 8,192 才对应 8 次累积。
已完成的第一轮和已停止的 4+3 试跑采用 FP32 前向、允许 TF32；
后续模板已接入 BF16 autocast 前向，参数与训练状态继续保存 FP32。
按 microbatch=1,024 重算的 300 轮预算见
[默认值变更与耗时估算](../experiments/20261031-infra/20261031-ppo-small-paper300/config-changes/microbatch-1024/README.md)。
