# 实验记录

一级目录统一使用 `yyyymmdd-{实验名称}`，日期按北京时间计算；内部保留原记录日期和名称。

| 目录 | 放什么 | 当前内容 |
| --- | --- | --- |
| [20261031-infra/](20261031-infra/README.md) | 环境、模拟器、初始化、性能测试和训练试跑 | PPO 基础设施准备与验证 |
| [20261031-scripted-baseline/](20261031-scripted-baseline/README.md) | 独立参考策略评测 | 脚本机器人 TavernBench 基线 |
| [20261031-exp01-ppo-scripted/](20261031-exp01-ppo-scripted/README.md) | **第一次正式模型实验**的训练与评测 | 100 PPO × 32 minibatch；整局测试和 TavernBench |
| [20261031-exp02-mixed-opponents/](20261031-exp02-mixed-opponents/README.md) | **第二次正式模型实验** | 2 V1 + BF16 + 独立 Actor 已启动；旧 4 V1 尝试停止并保留 |

查第一次正式实验，直接进入 **[20261031-exp01-ppo-scripted](20261031-exp01-ppo-scripted/README.md)**。
安装验证、旧 300 轮预演和 100 次 Adam 小试跑都归入 infra，不占正式实验编号。

每个日期目录保留 README、配置、状态、原始指标、评估结果和必要的复现脚本。
同一正式实验的训练和后续 benchmark 放在同一个 expXX 组；不同训练方案另开下一组。

## 模型与日志

- experiments 存记录，runs 存模型；本次只整理记录目录，模型的目录、文件和内容不变。
- 模型继续放在 `runs/yyyymmdd-{exp-name}/` 并由 Git 忽略；后续 best 名称使用北京时间前缀。
- 新实验显式指定 `--experiment-dir experiments/20261031-exp02-mixed-opponents/yyyymmdd-{exp-name}`，
  `--run-dir runs/yyyymmdd-{exp-name}`；两者不要求目录深度相同。
- 后续使用 W&B。配置、结果和指标 journal 保留；SDK 缓存、旧 TensorBoard、控制台、
  sessions、reruns、实时进度和进程文件留在本地并忽略。

目录迁移对照见 [layout-migration.json](layout-migration.json)。
历史命令、provenance 和答案封存清单记录运行当时的路径／源码摘要，原样保留；
旧路径可查迁移对照。当前 README 链接和复现脚本使用新路径。

## 默认脚本机器人说明

默认单智能体环境中的七个机器人对手，共用同一套 `crates/taverngym/src/bot.rs` 决策代码和启发式参数。八机器人演示与吞吐量测试则由这同一套策略控制全部八个席位；没有为各席位配置不同风格、不同模型或独立学习过程。

各席位的英雄、商店、手牌、阵容、金币及随机事件不同，因此具体行动会不同。随机选择使用游戏的随机数流，并非每个机器人各有一套不同策略；在输入状态和随机数状态相同时，决策是确定的。规则导致的局内成长不属于策略参数学习。

论文中的七个经过现实校准的对手席位，逐回合从按真实对局校准的对手库中取出阵容、手牌及隐藏计数器；它们与上述默认脚本机器人不是同一类对手配置。解释实验结果时，应区分“相同启发式的多个席位”“不同策略的对手群”和“校准的库对手”。

依据：[机器人实现](../crates/taverngym/src/bot.rs)、[单智能体环境](../crates/taverngym/src/env.rs)、[论文第 6 节](https://arxiv.org/html/2610.33845v1#S6)。
