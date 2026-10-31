<a name="top"></a>

<p align="center">
  <img src="docs/assets/autotavern-banner.png" alt="AutoTavern: fantasy tavern, tactical battlefield and AI decision trees" width="100%">
</p>

<h3 align="center">TavernGym &amp; TavernBench</h3>

<p align="center">
  A fast simulator and a decision benchmark for Hearthstone Battlegrounds.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-cb963e?style=flat-square" alt="License: MIT"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3572a5?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 or newer"></a>
  <a href="rust-toolchain.toml"><img src="https://img.shields.io/badge/Rust-1.98-995c3c?style=flat-square&amp;logo=rust&amp;logoColor=white" alt="Rust 1.98"></a>
  <a href="tavernbench/data"><img src="https://img.shields.io/badge/TavernBench-960_tasks-237f89?style=flat-square" alt="TavernBench: 960 tasks"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#taverngym">TavernGym</a> ·
  <a href="#terminology">术语与计数</a> ·
  <a href="#tavernbench">TavernBench</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="#citation">Citation</a>
</p>

## Overview

AutoTavern brings the decisions of Battlegrounds into a reproducible environment for AI
research: simulate complete games, connect a learning algorithm, or evaluate a model's
judgment against a fixed set of tasks.

| [TavernGym](docs/TAVERNGYM.md) | [TavernBench](docs/TAVERNBENCH.md) |
| :--- | :--- |
| A deterministic simulator for complete eight-player games. | A benchmark for reasoning about game decisions and rules. |
| 978 game objects from patch 36.6.3, with [content coverage](docs/CONTENT.md). | 960 tasks in 40 scenarios across eight categories. |
| Rust engine, Python bindings, single-seat and batched environments. | Prompts, labels, saved states, model answers, scoring and tool-use evaluation. |

This is the code and data release for *Build, Don't Play: Self-Improving LLM Research Agents
Develop a Hearthstone Battlegrounds AI That Rivals Top Players*, by Changlian Ma and Limin
Wang, Nanjing University and Shanghai AI Laboratory.

The release includes TavernGym and TavernBench. The trained TavernAgent, its weights and
training code, and the benchmark item generators are not included.

## Quick start

```bash
git clone https://github.com/MCG-NJU/AutoTavern.git
cd AutoTavern
```

### Run a game

Install with Python 3.10 or newer and a [Rust toolchain](https://rustup.rs). The version in
[`rust-toolchain.toml`](rust-toolchain.toml) is installed automatically by rustup.

```bash
pip install .
```

```python
import taverngym

game = taverngym.Game(seed=1)
while not game.is_over():
    for seat in range(game.num_players):
        if game.needs_action(seat):
            game.step(seat, game.bot_action(seat))

print(list(game.placements()))
```

This plays a full eight-player game with the included scripted bots. Replace a bot's action
with your own legal action to control a seat.

### Score the released answers

From the repository root:

```bash
cd tavernbench
python -m tavernbench.report --set data/eval --runs results/eval \
    --systems tavernagent,gpt-6-astra,claude-opus-5.5,jev
```

This scores released answers against the labels. It needs only Python's standard library: no
simulator build, model access or API key is required.

## TavernGym

TavernGym reproduces solo Battlegrounds for client **36.6.3.253932**, server rules **36.6.3**.
It exposes the player-visible state, an exact legal action mask, and seeded save and restore.

| Interface | Use it for |
| :--- | :--- |
| `Game` | Drive any seat, inspect a game, or branch from a saved state. |
| `BattlegroundsEnv` | Connect one learning seat against seven scripted bots. |
| `BatchEnv` | Step many games together, with any set of controlled seats, including self-play. |

The learning interface provides 245 discrete actions, 3,227 numeric features and 124 card
indices per observation. Rules and lobby settings can be configured explicitly.

[Environment guide](docs/TAVERNGYM.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Implemented cards](docs/CONTENT.md) · [Timewarp rules](docs/TIMEWARP.md)

<a id="terminology"></a>

## 术语与计数

本项目统一区分**游戏回合（turn）**、**玩家操作（action）**、**PPO 迭代（iteration）**和
**优化器更新（optimizer step）**；汇报时不单独用“轮”或“step”代替这些单位。

### 游戏与模拟器

一个完整八人局（lobby/game）包含多个游戏回合（turn）；每回合包含招募操作和自动战斗。

| 中文名称 | 英文术语 | 本项目中的含义 |
| --- | --- | --- |
| 完整八人对局 | Lobby / Game | 从选英雄开始，到决出第一名；默认 8 个玩家席位。 |
| 玩家席位 | Seat | 一名玩家的位置，可由脚本、固定模型或学习者控制；同一模型可以控制多个独立席位。 |
| 游戏回合 | Turn | 第 1、2、3……回合；招募阶段结束后自动结算战斗，再进入下一回合。 |
| 招募阶段 | Recruit phase | 买卖、出牌、升本、发现、选择目标、调整站位等操作所在的阶段。 |
| 单场战斗 | Combat | 两方阵容的一次自动交战；一个游戏回合可有多场战斗。 |
| 战斗攻击轮次 | Attack turn | 战斗内部的一次攻击推进，不是玩家操作；风怒等还可能产生多个攻击或效果事件。 |
| 玩家操作 | Action | 一次实际执行的离散动作，例如购买、移动、选择目标、冻结或 EndTurn。 |
| 玩家决策 | Decision | 策略根据当前观测选择一个动作；前向计算可以同时批量处理多个玩家决策。 |
| 合法动作掩码 | Legal action mask | 当前允许选择的动作集合。动作空间共 245 个编号，不表示每回合可操作 245 次。 |
| 待处理选择 | Pending choice | 英雄、发现、目标等尚未解决的选择；通常会阻塞其他操作。 |
| 玩家对局轨迹 | Episode | 学习者从入场到淘汰或获胜。玩家淘汰时，完整八人局可能尚未结束。 |
| 玩家对局长度 | Episode length | 该 episode 中学习者实际执行的操作数，不包含对手动作和战斗内部攻击。 |

当前训练在学习者淘汰或获胜后就重置该环境，不继续等待其余玩家决出第一名。
因此日志 `games_finished` 是**已完成的学习者 episode 数**；`actions_per_game` 是这些 episode
的平均学习者操作数，不是完整八人局中所有玩家的操作之和。

### 操作数、回合数与保护上限

以下为 **2026-10-31 实验 03** 的配置及其继承的原生默认值，不能当作所有实验的通用设置。

| 名称 | 专业术语／字段 | 当前数值与行为 |
| --- | --- | --- |
| 每回合操作计数 | Actions per turn；`actions_this_turn` | 每个玩家单独累计，新游戏回合开始时归零。选择、取消、EndTurn 也计数；被拒绝或延期未执行的动作不计数。 |
| 每回合操作上限 | Per-turn action cap；`turn_action_caps` | 当前 `[60]`，表示所有游戏回合使用阈值 60；达到后只允许 EndTurn 和解决待处理选择。 |
| 操作阈值的总上界 | `max_actions_per_turn` | 默认 200，只限制 `turn_action_caps` 表中阈值的上界；空表表示不启用操作上限，不能仅凭此字段认为存在 200 次限制。 |
| 整局回合保护上限 | Game turn limit；`max_turns` | 默认 100 回合；触发后，存活玩家按生命值与护甲之和排序，同分按席位排序。 |
| 战斗攻击保护上限 | Combat attack cap；`max_combat_attacks` | 默认 400 次攻击推进；触发后作为模拟器截断平局报告，不作为正常战斗胜负。 |
| 效果嵌套保护上限 | Effect resolution depth；`max_effect_depth` | 默认 64 层，用于防止效果解析无限递归，不是玩家操作预算。 |
| 整个 episode 的步数限制 | Episode step limit / TimeLimit；通常为 `max_episode_steps` | 与单回合上限不同；当前训练没有把整局固定切成 1,024 步。 |

**60 是允许继续普通操作的阈值，不是所有动作记录的严格总上限。**
EndTurn 本身也计数，待处理选择仍可以继续解决，所以单个回合记录的动作数可能超过 60。
这是模拟器的操作预算；它不同于真实客户端按秒计时的回合限制，也不同于 PPO 的采样长度。
计数与限制的依据见 [动作实现](crates/taverngym/src/actions.rs) 和 [规则配置](crates/taverngym/src/config.rs)。

`EndTurn` 是编号 0 的动作。模型可在其他动作仍合法时主动选择它；达到操作上限后，
在没有待处理选择的情况下，掩码只保留 EndTurn，属于被上限强制结束。
这会继续同一局游戏，不是结束整个 episode；不要把它直接称为 episode truncation。
整局终止与时间／步数截断的通用区别见 [Gymnasium 文档](https://gymnasium.farama.org/v1.2.3/tutorials/gymnasium_basics/handling_time_limits/)。

### 采样与 PPO 训练

以下仍以实验 03 为例：**2 脚本 + 2 固定 V1 + 3 固定 V2 + 1 学习者**，模型均随机采样、
BF16 前向。固定策略只参与对战，只有学习者的实际操作进入训练样本。

| 名称 | 专业术语／字段 | 当前定义与数值 |
| --- | --- | --- |
| 并行环境数 | Number of environments；`n_envs` | 128 个八人局环境。 |
| 学习者样本 | Transition / Sample | 一次实际学习者操作形成的观测、动作、奖励、下一观测、结束标志等记录。 |
| 每环境采样长度 | Rollout length / Rollout horizon；`n_steps` | 每次 PPO 收集 1,024 个学习者样本，可以跨多个游戏回合和多个 episode。 |
| 一批新采样数据 | Rollout batch | `128 × 1,024 = 131,072` 个新样本，之后才进入 PPO 优化。 |
| 有效训练批次 | Minibatch；`batch_size` | 每批 8,192 个样本，每个 epoch 有 16 批。 |
| 梯度累积的小块 | Microbatch；`micro_batch_size` | 每块 1,024 个样本，分别前向和反向；累积 8 块后执行一次 Adam。 |
| 数据重复训练次数 | Epoch；`n_epochs` | 当前为 1，正常情况下每批 rollout 数据训练一遍。 |
| 一次权重更新 | Optimizer step / Adam step | 当前每个有效 minibatch 对应一次；KL 提前停止时，未执行的更新不计数。 |
| 一次采样与优化周期 | PPO iteration | 收集新 rollout，再做最多 16 次 Adam；当前共计划 200 次 PPO。 |
| 推理批次 | Inference batch | 同一策略的就绪请求合批，与训练 minibatch 不同；本配置学习者、固定 V1、固定 V2 单次最多分别为 128、256、384 个请求。 |
| 原生批量推进调用 | Batched environment step；`BatchEnv.step()` | 可推进多个环境及多个受控席位，不等于一个学习者样本或一次游戏回合。 |

所有角色的采样前向均不构建梯度图；PPO 更新时只对学习者重新前向和反向。
同一固定版本的多个席位共用一份冻结模型，不与学习者共享可更新参数。

本阶段预算为 `200 × 128 × 1,024 = 26,214,400` 个新样本，最多 `200 × 16 = 3,200` 次新 Adam 更新。
继承 V2 的 Adam 历史计数 3,200，阶段内日志从 0 开始；累计计数最多到 6,400。
W&B 横轴 `optimizer_step` 是阶段内实际 Adam 次数，`train/adam_total_steps` 另记继承历史后的累计数。

### 指标、奖励与上限如何选择

| 指标 | 含义 |
| --- | --- |
| `cap_end_rate` | 每个完成 episode 内，因操作上限强制 End 的回合数／所经历的招募回合数，再对 episode 求平均；不是胜率，也不是全部动作中 End 的比例。 |
| `cycle_returns` | 每个完成 episode 返回已见完整可见状态的次数；本轮旧模式只覆盖 Move／Freeze 片段，新模式 `all_actions` 覆盖同回合所有非 End 动作；不代表所有无效操作。 |
| `actions_per_game` | 每个完成学习者 episode 的实际操作数均值，包括主动 End 和强制 End。 |
| `operation_penalty` | 当前每次 Move／Freeze 扣 `0.0001`；买卖、出牌、刷新等不收这项成本。 |
| `cycle_penalty` | 当前每次检测到无进展状态重复额外扣 `0.0002`。 |
| `total_reward` | 名次奖励加上述两项惩罚，实际用于 GAE 与 PPO；`cap_end_rate` 本身没有直接扣分。 |

不能把 `1 - cap_end_rate` 直接当成模型主动 End 的比例；淘汰、跳过回合等也会影响统计。
循环次数下降也不必然意味着操作数或触顶率下降，例如连续尝试新的站位仍可能耗尽预算。
后续训练模板已按确认改为“只罚重复”：`operation_cost=0`、`cycle_cost=0.0002`，
并补齐出牌／选择／取消等循环检测，显式配置 `cycle_detection="all_actions"`；实验 03 仍沿用旧奖励。
旧配置缺少此字段时保留原规则，不更改当前训练。详细口径见 [训练奖励配置](docs/TRAINING.md)。

每回合操作上限没有统一标准，60 是本地实验选择。选择时应结合真实客户端回合时间、
正常策略的操作耗时和每回合操作数分布，并按回合阶段、英雄／构筑查看 P50／P90／P95／P99。
当前大量回合被 60 截住，观测到的分布受上限限制，不能用它证明 60 足够。
应在独立、有界的评估中比较不同预算下的名次、有效操作、重复与耗时，避免让保护上限代替模型学习何时结束。
比较结果必须同时说明对手数量与版本、随机／贪心模式及评估局数。

当前参数以 [实验 03 配置](experiments/20261031-exp03-v2-continuation/20261031-ppo-v3-2v1-3v2-warmup/config.json)
为准；更详细的操作入口见 [训练说明](docs/TRAINING.md)、
[PPO 批次对照](docs/PPO_BATCH_COMPARISON.md) 和 [指标说明](docs/TRAINING_METRICS.md)。

## TavernBench

TavernBench tests combat predictions, warband arrangements, target choices and turn plans
with a shared prompt format and common scoring code.

| Category | Examples |
| :--- | :--- |
| Combat | Predict outcomes, damage and lethal risk. |
| Positioning | Arrange a warband against shields, cleave and deathrattles. |
| Ordering | Sequence plays, spells and triggers. |
| Targets | Choose a spell, Hero Power or minion target. |
| Selection | Choose a hero, trinket, discovery or play. |
| Strategy | Choose a turn plan: upgrade, buy, refresh or save the Gold. |
| Economy | Reason about Gold, purchases and upgrade costs. |
| Rule execution | Apply game rules to concrete actions. |

The **368-task evaluation set** is released alongside all **960 tasks**. Saved simulator
states cover the 30 play scenarios; the ten rule-execution scenarios need no state. The
released answers include TavernAgent, ten language models, Jev, and five tool-use runs.

To evaluate your own model, install the optional runners from the repository root:

```bash
pip install ".[bench]"
```

[Benchmark guide](docs/TAVERNBENCH.md) · [Task data](tavernbench/data) ·
[Released answers](tavernbench/results) · [Scoring code](tavernbench/tavernbench/score.py)

## Documentation

| Guide | Contents |
| :--- | :--- |
| [术语与计数](#terminology) | 游戏回合、操作、episode、保护上限、rollout、PPO 与 Adam 的中英定义及当前配置。 |
| [TavernGym](docs/TAVERNGYM.md) | Game APIs, batched environments, observations, actions, configuration and speed. |
| [TavernBench](docs/TAVERNBENCH.md) | Dataset layout, labels, scoring, model runners, tool use and reproduction notes. |
| [Architecture](docs/ARCHITECTURE.md) | Engine structure, interfaces and how the components fit together. |
| [Content coverage](docs/CONTENT.md) | Implemented minions, heroes, spells, trinkets and other game objects. |
| [Timewarp](docs/TIMEWARP.md) | The Timewarped Tavern (off by default in 36.6.3; two heroes still visit it). |
| [PPO setup and small model](docs/TRAINING.md) | uv, Python 3.12, initialized entity PPO, training entry point, and model and log locations. |
| [Training infrastructure](docs/TRAINING_INFRA.md) | Lossless rollouts, gradient accumulation, checkpoint recovery, profiling, precision and Git artifact policy. |
| [PPO batch comparison](docs/PPO_BATCH_COMPARISON.md) | Rollouts, minibatches, epochs, optimizer steps and training budgets compared with reference implementations. |
| [Training metrics and W&B](docs/TRAINING_METRICS.md) | Per-Adam train curves, step-zero validation, Weave setup and migration without restarting training. |
| [训练方法论与 agent 协调](docs/TRAINING_METHODOLOGY.md) | 历史对手池、冠军晋级、持续监控、恢复和停止条件；当前为未启用的流程与角色草案。 |
| [Experiment records](experiments/README.md) | Dated experiments, reproduction commands, environment details and raw results. |

For development, run `pip install -e ".[test]"` and `pytest`. The Rust crate also works on its
own with `cargo test --release`.

## Citation

If you use TavernGym or TavernBench in your research, please cite:

```bibtex
@misc{ma2026build,
  title  = {Build, Don't Play: Self-Improving {LLM} Research Agents Develop a Hearthstone
            Battlegrounds {AI} That Rivals Top Players},
  author = {Ma, Changlian and Wang, Limin},
  year   = {2026},
  url    = {https://github.com/MCG-NJU/AutoTavern}
}
```

## License

Code, documentation and the TavernBench tasks, labels, states and results are released under
the [MIT License](LICENSE). Third-party game content retains its original rights; see
[NOTICE](NOTICE).

This is an independent, non-commercial research project, unaffiliated with Blizzard
Entertainment. Hearthstone and Hearthstone Battlegrounds are Blizzard trademarks. No game
client files or official game artwork are distributed, and TavernGym does not connect to
Blizzard's services. The AutoTavern banner and emblem are original generated illustrations.

<p align="center">
  <a href="#top"><img src="docs/assets/autotavern-emblem.png" alt="AutoTavern emblem" width="76"></a>
  <br>
  <a href="#top">Back to the tavern</a>
</p>
