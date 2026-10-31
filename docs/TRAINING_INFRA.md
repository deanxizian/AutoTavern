# 训练基础设施改进记录

日期：2026-10-31，Asia/Shanghai。

本轮为 TavernGym 增加可初始化、保存和继续训练的实体 PPO，以及可复核的实验与性能记录。
旧 300 轮训练按用户要求停止，保留三个完整 PPO 迭代的检查点。
后续首版改为 100 个有效 minibatch、100 次 Adam 更新，学习率 3e-4→3e-6 线性衰减，
配置和结果见 [首版实验](../experiments/20261031-infra/20261031-ppo-small-v1-100-minibatches/README.md)。
已有性能测试没有改写这些权重。论文原始 TavernAgent 的权重、完整训练配方和校准对手库不包含在本实现中。

## 已实现的改进

| 项目 | 实现与作用 | 代码／记录 |
| --- | --- | --- |
| 可重复环境 | uv、Python 3.12、锁定的训练依赖组；基础模拟器保留 numpy-only 接口 | `pyproject.toml`、`uv.lock`、`.python-version` |
| 小模型初始化 | d=192、4 层、8 头、FFN=1024，共 3,491,978 个可训练参数；共享策略／价值／名次辅助主干 | `python/taverngym/learning/model.py`、`initialize.py` |
| 卡牌表示 | 身份嵌入、静态特征投影、固定版本 MiniLM 规则文本向量投影；缓存校验卡牌词表和数据摘要 | `text.py`、`policy.py` |
| 原生批量采样 | 单个受控席位与七个同策略脚本对手；native BatchEnv 提供多线程、精确合法动作掩码、自然终局及自动重置 | `vector_env.py` |
| 无损 rollout 缓存 | 数值保持 FP32，二值字段按位打包，卡牌 ID 用 uint16；只解包当前 minibatch，不复制整个 rollout | `buffer.py` |
| 梯度累积 | 按完整有效 batch 归一化优势和名次损失；逐 microbatch 反向，完整 minibatch 后裁剪并执行一次 Adam | `ppo.py` |
| 学习率与预算 | 按全局采样进度线性调度学习率，区分 microbatch、minibatch、PPO 迭代；总预算现为 300 轮 | `schedule.py`、`experiment.py` |
| 检查点恢复 | 恢复完整轮检查点的权重和 Adam 状态，追加指标，分开保存续训日志；拒绝从半轮 rollout 计数恢复 | `experiment.py --resume` |
| 评估随机流隔离 | 固定评估种子、每席位对局配额；保存／恢复 Torch RNG，避免评估推进训练随机流 | `experiment.py` |
| 性能与数值检查 | 分开记录模拟器、推理、传输和缓存写入；固定权重、Adam 和样本比较 microbatch 与精度 | `experiments/20261031-infra/20261031-microbatch-scaling/`、`20261031-bf16-mixed-precision/` |
| 实验与模型分离 | experiments 保存配置、指标和结论，runs 保存初始化／最近／最佳／定期／最终模型 | [目录与命令](TRAINING.md) |
| W&B 与 Weave | 每次 Adam 记录 train，初始评估作为 val 的 step=0；当前训练不重启，通过旁路补录后持续同步 | [指标与迁移](TRAINING_METRICS.md)、`tracking.py`、`migrate_wandb.py` |

断点续训会重新初始化环境与 RNG，不是逐游戏、逐位一致的继续执行。
名次标签仅作为已结束片段的训练目标，不进入玩家可见观测。
当前公开观测还有附着物标识、部分持久效果与对手战斗历史细节的缺失；本实现没有通过内部 state_json 绕过信息边界。

## 缓存与批次

每条观测从 13,649 字节降至 **3,306 字节**，约 4.13 倍无损压缩：
677 个 FP32 数值维、2,550 个二值维、124 个卡牌 ID、245 个动作合法性位。
完整一轮观测约 12.914 GiB，加奖励、概率、值、优势和标签后约 **13.07 GiB CPU RAM**。
仅保留当前轮并循环复用，300 轮累计约 3.78 TiB 观测处理量不是同时驻留或永久落盘的数据量。
可选磁盘缓存保留相同采样语义。

| 单位 | 样本量 | microbatch 数 | Adam 更新次数 |
| --- | ---: | ---: | ---: |
| 默认 microbatch | 1,024 | 1 | 每 8 个执行一次 |
| 有效 minibatch | 8,192 | 8 | 1 |
| 一轮 PPO，epoch=1 | 4,194,304 | 4,096 | 最多 512 |
| 300 轮，统一使用新默认 | 1,258,291,200 | 1,228,800 | 最多 153,600 |

默认值 **1,024** 已写入新建 EntityPPO、初始化模板及两个训练入口。
正式实验配置显式指定 1,024；通用训练入口通过 `--micro-batch-size` 覆盖旧检查点内的值。
默认大小是切片上限，实际 minibatch 更小时只使用该 minibatch 的样本数。
前三轮历史值为 256，历史运行时配置和实测结果保留原值，不回写成 1,024。

## 精度与性能

已完成的第一轮和已停止的 4+3 试跑使用 FP32 前向，并设置
`torch.set_float32_matmul_precision("high")`，即 **允许 TF32 的 FP32 基线**，不是严格 IEEE FP32 矩阵计算。
没有逐 kernel 确认所有算子实际使用 TF32。

后续已接入 `forward_precision=bf16`：网络前向 autocast 为 BF16，head 输出转回 FP32 后计算概率、
log-prob、ratio、KL 和损失，参数、梯度累积及 Adam 保持 FP32。
学习者采样、训练重算、bootstrap value 和验证共用该入口，固定对手缺省跟随同一精度。
旧文件缺省仍是 FP32，模式会随新检查点保存／加载，不转换已有权重文件。
接入与验证见 [BF16 训练配置](../experiments/20261031-infra/20261031-bf16-training-integration/README.md)。

此前 RTX 5080 的同模型、同数据、同 Adam 状态补测，每档预热后计时 9 次，取同步墙钟中位数：

| microbatch | 允许 TF32 的 FP32 更新 ms | BF16 混合更新 ms | FP32 峰值 GiB | BF16 峰值 GiB |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 657 | 501 | 1.45 | 1.09 |
| 512 | 634 | 451 | 2.68 | 1.97 |
| 1,024 | 663 | 415 | 5.12 | 3.74 |
| 2,048 | 633 | 442 | 10.04 | 7.26 |

每次更新都处理 8,192 条样本。batch 已驻留 GPU，不含环境、CPU 解包和 H2D；
显存是测试进程 PyTorch 峰值 allocated，不是桌面和 CUDA 上下文在内的整卡占用。
1,024 是本次推荐的默认工作点；短时计时有波动，不能将单次排名解释为稳定最优。
BF16 数值检查中梯度均有限，但相对 L2 差约 8.9%，尚未验证长时间收敛和胜率。

完整 300 轮的外推预算为：当前精度约 **52–54 小时**，BF16 方案约 **39–42 小时**。
外推结合已有三轮完整耗时、扣除 profiling 暂停，再按固定 batch 相对速度缩放；
不包含额外评估、模型保存和中断，也不等于完整 BF16 训练实测。
依据和重算脚本见 [预算变更记录](../experiments/20261031-infra/20261031-ppo-small-paper300/config-changes/microbatch-1024/README.md)。
CPU 与 GPU 当前按数据依赖串行执行，没有实现跨采样／更新阶段的流水并行。

混合对手的独立 Actor 采样已有单独原型：256 局、6 脚本 + 1 固定 V1 + 1 学习者，
两个策略分别用 BF16 合批，让各局独立推进。两个短测种子合计 65,536 个学习者样本，
采样从 55.502 秒降至 10.076 秒，约 5.51 倍；固定 V1 batch 中位数从 2 增至 115。
该数字是当时的采样原型测量；后续已接入 PPO 缓存／GAE 并启动第二轮，完整首轮预检采样 41.56 秒、更新 17.25 秒；
配置、计时和边界见 [Actor 采样记录](../experiments/20261031-infra/20261031-unified-actor-sampling/README.md)。

## Git 产物约定

| 提交到 Git | 留在本地并忽略 |
| --- | --- |
| 源码、测试、默认配置、uv.lock、.python-version | `.venv/`、构建目录、包和下载缓存 |
| 实验 README、配置快照、metrics/results/summary、验证及来源摘要 | `runs/` 内所有模型、文本向量和检查点 |
| 性能基准原始 JSON、脚本机器人答案／评分及复现脚本 | TensorBoard、W&B SDK 缓存、sessions、实时 process/progress/launcher/training 文件 |
| 有审查价值的有限原始测试文本结果 | 控制台／编译／stderr 日志、完整游戏事件流、重复 legacy-runs 和未筛选 reruns |
| 论文来源索引 | 本地论文 PDF、源文件、译文、HTML 预览及图片 |

`.gitignore` 仅排除上述生成物，不忽略整个 experiments 或全部 JSON，避免把结果证据一起隐藏。
忽略不会删除文件。新克隆缺少本地权重、下载文件和实时进度文件属于预期行为；
论文原始来源见 [论文索引](papers/README.md)，初始化和训练命令见 [训练说明](TRAINING.md)。

## 验证

学习测试覆盖：三个输出头及梯度、精确合法动作掩码、观测信息边界、保存／加载、
已结束片段的名次监督、真实观测位级无损压缩、dense／disk／compact 更新等价、
梯度累积等价、多线程采样确定性、自然终局、调度端点、评估 RNG 隔离和检查点续训。

```bash
uv run --locked --group train --extra test python -m pytest -q
```

基础模拟器 CI 没有安装 train 依赖时，会跳过学习测试；本地完整验证应保留 `--group train`。
初始化生成真实规则文本缓存需要下载固定版本 MiniLM；测试使用合成向量，不依赖模型 API 或外部推理服务。

本次整理后的本地完整 Python 验证为 **59 passed**；`uv lock --check --offline` 通过。

独立 Actor 的缓存次序、GAE、奖励指标、两轮更新和 W&B 接入验证见
[接入记录](../experiments/20261031-infra/20261031-actor-ppo-integration/README.md)，当前完整 Python 测试为 99 passed。
