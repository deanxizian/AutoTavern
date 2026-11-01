# 实验记录

**查看当前效果、进度、最佳 checkpoint 和最终对战表：[实验总览](EXPERIMENT_OVERVIEW.md)。**
这里是目录导航；每次实验完成或停止时同步更新总览，原始配置与结果保持封存。

一级目录统一使用 `yyyymmdd-{实验名称}`，日期按北京时间计算；内部保留原记录日期和名称。

| 目录 | 放什么 | 当前内容 |
| --- | --- | --- |
| [20261031-infra/](20261031-infra/README.md) | 环境、模拟器、初始化、性能测试和训练试跑 | PPO 基础设施准备与验证 |
| [20261031-scripted-baseline/](20261031-scripted-baseline/README.md) | 独立参考策略评测 | 脚本机器人 TavernBench 基线 |
| [20261031-exp01-ppo-scripted/](20261031-exp01-ppo-scripted/README.md) | **第一次正式模型实验**的训练与评测 | 100 PPO × 32 minibatch；整局测试和 TavernBench |
| [20261031-exp02-mixed-opponents/](20261031-exp02-mixed-opponents/README.md) | **第二次正式模型实验** | 2 V1 + BF16 + 独立 Actor 已完成；含 TavernBench 与 2 脚本 + 3 V1 + 3 V2 对战 |
| [20261031-exp03-v2-continuation/](20261031-exp03-v2-continuation/README.md) | **第三次正式模型实验** | 从 V2 保留 Adam；128 环境，2 脚本 + 2 V1 + 3 V2；warmup 峰值 1e-5 |
| [20261031-exp04-v4-cycle-only/](20261031-exp04-v4-cycle-only/README.md) | 第四次正式实验及中断尝试 | 正式100 PPO完成；best为PPO0，废弃→V3；软循环扣分、60步只End |
| [20261031-exp05-v5-cycle-forfeit/](20261031-exp05-v5-cycle-forfeit/README.md) | 第五次正式实验及独立对战 | 100 PPO完成，保留PPO100；含两轮320局对战与逐回合诊断 |
| [20261031-exp06-v6-cycle-forfeit/](20261031-exp06-v6-cycle-forfeit/README.md) | 第六次正式实验 | 100 PPO完成；best为PPO0，废弃→V5 |
| [20261101-exp07-v7-history-ranked/](20261101-exp07-v7-history-ranked/README.md) | V7旧排名现场与新移动成本准备 | 旧排名中止、训练0；新稿直接继承V5，move_cost=0.0005，未启动 |
| [20261101-exp08-v8-history-ranked/](20261101-exp08-v8-history-ranked/README.md)／[exp09](20261101-exp09-v9-history-ranked/README.md) | 未启用的旧条件草案 | V8/V9均未训练，不自动推进 |

查第一次正式实验，直接进入 **[20261031-exp01-ppo-scripted](20261031-exp01-ppo-scripted/README.md)**。
安装验证、旧 300 轮预演和 100 次 Adam 小试跑都归入 infra，不占正式实验编号。

每个日期目录保留 README、配置、状态、原始指标、评估结果和必要的复现脚本。
同一正式实验的训练和后续 benchmark 放在同一个 expXX 组；不同训练方案另开下一组。

## 实验版本与配置

有效训练版本清单见 [model-versions.json](../configs/model-versions.json)：V4→V3、V6→V5为废弃别名，
后续新对手池排除，旧权重与历史记录保留；V7新准备直接继承V5。

2026-11-01 新准备：[V7 独立移动成本 0.0005](20261101-exp07-v7-history-ranked/20261101-ppo-v7-move0005-cycle001-forceend/README.md)。
循环费用仍0.01、60步只强制End；尚未启动训练。此前中断的V7及V1–V6记录保持原样。

各版本使用自己的完整配置；通用初始化模板不是旧实验的配置来源。
新方案准备在新目录，已有实验的配置和 provenance 不随新版本改写。

| 版本 | 独立配置 | 奖励方案 | 阶段 |
| --- | --- | --- | --- |
| V1 | [exp01 配置](20261031-exp01-ppo-scripted/20261031-ppo-small-100ppo-32minibatches/config.json) | 原始名次奖励 | 已训练版本 |
| V2 | [exp02 配置](20261031-exp02-mixed-opponents/20261031-ppo-mixed-2v1-bf16-actor/config.json) | 名次 + Move／Freeze 成本 + 连续片段循环成本 | 已训练版本 |
| V3 | [exp03 配置](20261031-exp03-v2-continuation/20261031-ppo-v3-2v1-3v2-warmup/config.json) | 保留 V2 奖励；继承 Adam 并 warmup | 已完成 200 PPO；保留 PPO150 best |
| V4 | [正式exp04配置](20261031-exp04-v4-cycle-only/20261101-ppo-v4-1script-2v1-2v2-2v3-cycle001-forceend/config.json) | 名次 + 全动作返回每次扣0.01；60步只End | 完成100 PPO，保留PPO0，废弃→V3 |
| V5 | [正式exp05配置](20261031-exp05-v5-cycle-forfeit/20261101-ppo-v5-1script-1v1-1v2-2v3-2v4-cycle001-forceend/config.json) | 沿用V4软扣分 | 完成100 PPO，保留PPO100，当前V7来源 |
| V6 | [正式exp06配置](20261031-exp06-v6-cycle-forfeit/20261101-ppo-v6-1script-1v1-1v2-1v3-1v4-2v5-cycle001-forceend/config.json) | 沿用V4软扣分 | 完成100 PPO，保留PPO0，废弃→V5 |
| V7 | [新准备配置](20261101-exp07-v7-history-ranked/20261101-ppo-v7-move0005-cycle001-forceend/config.json) | 循环0.01 + 每次Move 0.0005 | 仅准备0 PPO，未启动，尚无自己的best |

每次正式启动使用本实验的 `--config`，记录源码 commit／脏工作区摘要、模型路径和 SHA-256。
通用代码保留旧模式兼容；不同配置对应不同训练规则，不需要为每个实验复制整套训练代码。

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

2026-11-01 起，用户将原生脚本机器人统一命名为 **V0**；V1 起为训练模型。
历史配置与接口中的 `scripted` 标识保留，新报告使用 V0。

本次独立对战：[V0／V1／V2／V3／V5 平衡八人局](20261031-exp05-v5-cycle-forfeit/20261101-match-v0-v1-v2-v3-v5/README.md)。
后续逐回合诊断：[同五版 320 局全量并行补测](20261031-exp05-v5-cycle-forfeit/20261101-match-v0-v1-v2-v3-v5-turn-diagnostics/README.md)，
保存存活招募回合、逐回合动作分布与可见状态返回统计，独立核验通过；旧比赛保持封存。

默认单智能体环境中的七个机器人对手，共用同一套 `crates/taverngym/src/bot.rs` 决策代码和启发式参数。八机器人演示与吞吐量测试则由这同一套策略控制全部八个席位；没有为各席位配置不同风格、不同模型或独立学习过程。

各席位的英雄、商店、手牌、阵容、金币及随机事件不同，因此具体行动会不同。随机选择使用游戏的随机数流，并非每个机器人各有一套不同策略；在输入状态和随机数状态相同时，决策是确定的。规则导致的局内成长不属于策略参数学习。

论文中的七个经过现实校准的对手席位，逐回合从按真实对局校准的对手库中取出阵容、手牌及隐藏计数器；它们与上述默认脚本机器人不是同一类对手配置。解释实验结果时，应区分“相同启发式的多个席位”“不同策略的对手群”和“校准的库对手”。

依据：[机器人实现](../crates/taverngym/src/bot.rs)、[单智能体环境](../crates/taverngym/src/env.rs)、[论文第 6 节](https://arxiv.org/html/2610.33845v1#S6)。
