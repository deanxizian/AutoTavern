# 实验记录

每次实验使用独立目录，命名格式为 `yyyymmdd-{exp-name}`。日期按 **Asia/Shanghai** 时区的实验开始日期计算；实验名使用简短的英文小写连字符形式。例如：`20261031-paper-reproduction`。

每个实验目录应保存：

- `README.md`：目的、方法、运行命令、结果和解释。
- `experiment.json`：实验标识、日期、状态、参数及产物索引。
- 原始输出与结构化结果，必要时附复现脚本、环境信息和代码版本。

实验进行时标记为 `in_progress`，完成后更新结论和状态。不同设置的独立实验另建目录；同日同类实验可以追加清楚的名称后缀。保留失败结果，并将原始测量与后续解释分开。

本目录用于保留可审查的实验记录；大规模临时产物、模型权重与构建缓存仍放在 `runs/`、`target/` 等目录。原始文本输出优先使用 `.txt`，避免被项目的 `*.log` 忽略规则排除。

Git 保留配置、实验说明、结构化指标／结果、基准原始样本及复现脚本。
TensorBoard、实时 progress/process/launcher/training 文件、逐次控制台日志、完整游戏事件流、
编译／stderr 输出、sessions、reruns 和重复 legacy-runs 留在本地并忽略；因此文档中的这些运行产物路径不一定存在于新克隆中。
`uv.lock` 和 `.python-version` 应提交。完整目录约定及本轮基础设施改进见
[训练基础设施](../docs/TRAINING_INFRA.md)。

| 实验 | 内容 |
| --- | --- |
| [20261031-scripted-bot-game](20261031-scripted-bot-game/README.md) | 八个脚本机器人完成一局游戏，保存事件日志、阵容与排名。 |
| [20261031-paper-reproduction](20261031-paper-reproduction/README.md) | 单核吞吐量复现与强化学习接口验证。 |
| [20261031-scripted-bot-tavernbench](20261031-scripted-bot-tavernbench/README.md) | 脚本机器人 TavernBench 基线，区分策略适配与战斗模拟器贡献。 |
| [20261031-ppo-library-setup](20261031-ppo-library-setup/README.md) | uv 与 Python 3.12 安装 PPO 库，验证 CUDA、模型存取与 TavernGym 接口。 |
| [20261031-ppo-small-init](20261031-ppo-small-init/README.md) | 初始化 d=192、4 层、3,491,978 参数的 PPO，保存零训练步数模型。 |
| [20261031-ppo-small-paper300](20261031-ppo-small-paper300/README.md) | 300 轮 × 4,194,304 决策；完成 3 轮后停止；默认 microbatch 改为 1,024，并记录两种精度的全量时间估算。 |
| [20261031-microbatch-scaling](20261031-microbatch-scaling/README.md) | 256／512／1024／2048 的前后向延迟、显存与有效 batch 更新耗时实测。 |
| [20261031-bf16-mixed-precision](20261031-bf16-mixed-precision/README.md) | 允许 TF32 的 FP32 基线与 BF16 autocast 对比；记录延迟、显存和 PPO 数值差异，正式训练保持停止。 |

## 训练模型与日志

训练产生的模型权重、定期检查点、最佳模型与最终模型，
统一放在 `runs/yyyymmdd-{exp-name}/`。`runs/` 已被 Git 忽略。
实验配置、说明、TensorBoard 日志、评估指标、原始文本输出和模型路径，
放在对应的 `experiments/yyyymmdd-{exp-name}/`，不将模型文件复制进实验记录。
PPO 模型沿用 SB3 原生 `.zip` 格式；具体目录和恢复说明见
[训练环境与产物说明](../docs/TRAINING.md)。

## 默认脚本机器人说明

默认单智能体环境中的七个机器人对手，共用同一套 `crates/taverngym/src/bot.rs` 决策代码和启发式参数。八机器人演示与吞吐量测试则由这同一套策略控制全部八个席位；没有为各席位配置不同风格、不同模型或独立学习过程。

各席位的英雄、商店、手牌、阵容、金币及随机事件不同，因此具体行动会不同。随机选择使用游戏的随机数流，并非每个机器人各有一套不同策略；在输入状态和随机数状态相同时，决策是确定的。规则导致的局内成长不属于策略参数学习。

论文中的七个经过现实校准的对手席位，逐回合从按真实对局校准的对手库中取出阵容、手牌及隐藏计数器；它们与上述默认脚本机器人不是同一类对手配置。解释实验结果时，应区分“相同启发式的多个席位”“不同策略的对手群”和“校准的库对手”。

依据：[机器人实现](../crates/taverngym/src/bot.rs)、[单智能体环境](../crates/taverngym/src/env.rs)、[论文第 6 节](https://arxiv.org/html/2610.33845v1#S6)。
