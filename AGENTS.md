# Repository Guidelines

## 项目结构

- `crates/taverngym/`：Rust 规则引擎、卡牌数据及集成测试。
- `crates/taverngym-py/`：PyO3 绑定；`python/taverngym/`：Python 环境封装，`learning/` 包含 PPO、采样、奖励和日志。
- `tests/`：Python 测试；`tavernbench/`：基准数据、模型适配与评分代码。
- `configs/`：配置；`docs/`：开发说明，`docs/assets/` 存放图片。
- `experiments/` 保存实验记录，`runs/` 保存模型，两者分开管理。

## 构建、测试与开发

使用 uv、Python 3.12，以及 `rust-toolchain.toml` 固定的 Rust 1.98.1。

```bash
uv sync --locked --python 3.12 --group train --extra test
uv run --locked --group train --extra test python -m pytest -q
cargo fmt --all -- --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release
uv run --locked --group train python -m taverngym.learning.experiment --help
```

上述命令依次安装依赖、运行完整 Python 测试、检查 Rust 格式和 lint、运行 Rust 测试、查看训练入口。
Rust 源码变更后，在安装命令追加 `--reinstall-package taverngym` 重建绑定。训练参数见 `docs/TRAINING.md`。

## 代码风格

Python 使用四空格缩进、`snake_case` 和 `PascalCase` 类名，遵循 `pyproject.toml` 的 Ruff 配置及 100 列设置。
Rust 使用 2024 edition，格式遵循 `rustfmt.toml`；原生接口变更须同步 Python 封装。

## 通用报告表格

2026-11-01 用户要求：今后向用户报告的可比指标表，按每列指标的优劣方向排名，
仅第一名用 Markdown 加粗（`**值**`），其他名次正常显示，适用于主 agent 和所有 subagent。
表头用 ↑／↓ 明确优劣方向；胜率／Top 1 等核心表现列靠前，再列效率、步数及行为指标。
按原始数值排名，并列第一同加粗；用户已取消第二名下划线，不使用 HTML 下划线标签。
未知／未记录值不排名；版本号、样本量和无明确优劣方向的描述列不标名次，不整行套用单一名次。
小数显示要足够区分真实差异；排版不改变原始指标、评估口径、置信度或选模规则。

## 测试约定

Python 测试命名为 `test_*.py`；Rust 测试位于模块内及 `crates/taverngym/tests/`。
未设置覆盖率百分比门槛；规则、合法掩码、终局及采样修复应补充固定种子的回归测试。
基础 CI 可能跳过训练依赖相关测试，完整验证须安装 `train` 组；报告 CUDA 测试是否实际执行。

## 实验、评估与配置

实验当前状态统一记录在 `experiments/EXPERIMENT_OVERVIEW.md`，指标摘要与原文件哈希见
`experiments/experiment-overview.json`。每次实验完成、停止、best选择或最终测评新增后同步更新，
清楚区分训练实际进度、所选checkpoint与其PPO/Adam、阶段val、收尾best独立测评及同场比赛；
标明对手/模式/局数，不能将不同阵容横向排名或把V7准备写成已训练。历史原记录不覆盖。

日期按北京时间，目录使用 `yyyymmdd-{exp-name}`，分组遵循 `experiments/README.md`。
最佳模型命名为 `YYYYMMDD-HHMMSS-best_model.zip`。模型权重、密钥和 SDK 缓存不得提交 Git。
续训只使用完整 PPO 轮检查点；保留失败现场，不改写历史配置和来源摘要。

2026-11-01 用户命名约定：原生脚本机器人统一称为 V0（`crates/taverngym/src/bot.rs`），
V1 起为训练模型。新报告使用 V0；旧配置与接口中的 `scripted` 标识保留兼容，不追溯改写历史。
“最强三个历史 bot 各两席 + 一个脚本”即三个训练版本各两席 + 一个 V0，V0 不占前三名额。

训练默认将指标与配置记录到 W&B 项目 `eric_eric_/AutoTavern`；train 横轴为实际 Adam 步数，起点评估放在 val 的 step=0。
每次汇报和表格须明确对手数量与版本、学习者随机／贪心模式及局数，例如“2 个固定 V1 + 5 个脚本，随机采样，64 局”。
后续混合实验显式设置 `evaluation_include_scripted=false`，验证只保留混合随机／贪心两组，
不额外跑七脚本对照；混合阵容中的脚本席位保留，历史运行的验证协议不改写。
后续新训练使用仅循环奖励：`operation_cost=0`、`cycle_cost=0.01`、
`cycle_detection="all_actions"`，覆盖移动／冻结、出牌／选择／取消与技能菜单往返；不改当前轮旧参数。
每次返回已见可见状态固定扣 0.01，总罚随次数线性累计，单次罚额不递增。
不因循环或操作预算提前判负；两种 forfeit 阈值均为 `null`，原生 60 步上限只强制 End，让对局继续。
W&B 记录每局重复次数、局均重复占比、连续返回峰值与累计循环罚分，并保留 Top 1／Top 4／平均名次。
这些新设置归入 V4 准备版，各实验使用自身的 `config.json` 与版本记录，勿用通用模板覆盖旧实验。
V4 入口为 `experiments/20261031-exp04-v4-cycle-only/`；准备配置不代表已授权启动训练。
2026-11-01 用户确认移动成本先用于 V7：新准备目录
`experiments/20261101-exp07-v7-history-ranked/20261101-ppo-v7-move0005-cycle001-forceend/`。
该 V7 拟定 `move_cost=0.0005`，`operation_cost=0`、`cycle_cost=0.01`，实际 Move 与循环费用可叠加；
旧 V1–V6 及已停止实例不追溯改写，缺 `move_cost` 按 0 保持兼容。只准备/验证，不启动训练或自动延伸 V8/V9。
V7 来源按用户最新指令直接使用 V5 best 与完整 Adam，跳过策略未更新的 V6。
版本生命周期清单为 `configs/model-versions.json`：V4→V3、V6→V5 标为 retired（废弃），
不进入后续新的开发参考池、排名候选或训练对手池，也不以不同版本名重复占席；旧引用仍可复现。
以后最终所选 best 的实际策略指纹与继承来源相同，就登记为废弃别名并指向有效原版，
不因版本号更高把它当成新冠军。需核对实际策略与完整 Adam，不能仅比 ZIP 字节或仅凭 PPO0 猜测。
废弃不代表未训练，也不删除历史模型、配置或日志。新实例将生命周期清单路径和排除版本封存入排名输入，旧实例保持原协议。
跨代方法见 `docs/TRAINING_METHODOLOGY.md`：从当前冠军继续、混入冻结历史冠军，统一晋级评估后才替换冠军。
后续对手阵容按 2026-11-01 用户确认：统一开发评估选最强的 3 个历史 bot，每版 2 席，加 1 个脚本；
学习者另占第 8 席。不再要求每个旧版本至少一席，也不保证最新版本必定进入对手池。
最强按同一参考混合池、种子、随机采样模式和局数评估的平均名次确定，不直接横比各阶段 val，
不使用独立 test 选池。未入当期池的历史权重保留，之后可重新参与评估；学习者来源仍按本轮约定的前阶段 best。
训练对手改变后的 val 不直接横比；保留历史池引用的旧冠军，自动循环须有明确预算与停止条件。
当前 best 仅按混合随机验证的 `val/mean_placement` 严格降低时替换；
first/top4、Greedy、循环与触顶率只作观察，独立 test 不选模。口径见 `docs/TRAINING_METRICS.md`。
基准推理不得读取标签，封存预测后再评分；使用已有官方评分器，测试集不参与选模。

## RSI agent 分工

用户要求 RSI、改进迭代或训练总控时，由当前主 agent 通过 subagent 工具调用
`.codex/agents/orchestrator.toml` 的 `orchestrator`，统筹诊断、方案、实施、正确性验证、
训练评估、版本留存和下一轮决策；可按需委派实施或专项复核 subagent。
进入需持续监控的训练阶段，或用户要求全程监控时，主 agent 拉起或复用
`.codex/agents/training-monitor.toml` 的 `training_monitor`，无需另建独立 chat。
先读取角色文件；若工具不能按角色名加载，读取其 `developer_instructions` 传给 subagent。
总控显式使用 `gpt-6.1-sol`、`high` 推理强度；监控使用 `gpt-6.1-terra`、`medium`，
立即检查并每 180 秒巡检。若工具不能按角色名加载模型配置，以不继承完整对话的方式
显式传入角色文件的模型、推理强度和指令，不沿用主 agent 的模型设置。
主 agent 给两个角色同一实例 ID、实际配置、已有授权、预算、停止条件和状态目录，
复用已有 subagent，统一接收结果和监控告警；由 orchestrator 统一决定链路推进与训练恢复。
RSI 每轮记录目标、假设、基线、修改、验证证据和采纳决定；正确性检查通过后才启动相关训练。
研究/实现轮数、训练阶段、样本与时间均受本轮预算约束；验证使用既定程序，必要时临时独立复核。
版本列表/范围、首阶段来源和各阶段预算由每次任务指定，不固定为某几个版本；
后续阶段自动继承前一阶段 best 的完整检查点，最后一个指定阶段及最终评估完成后停止。
仅配置/计划请求不启动训练；角色文件也不等于启动授权或已注册后台定时任务。
角色可安装副本与调用说明见 `configs/agents/README.md`。

## 测评 agent 与批量比赛

2026-11-01 用户要求专用测评角色：说“测评 Agent”“评测 Agent”或“比赛 Agent”时，
主 agent 读取 `.codex/agents/evaluator.toml`，通过 subagent 工具调用或复用 `evaluator`。
按用户要求固定使用 `gpt-6.1-terra`、`medium`，不新建独立 chat；传入来源、阵容、模式、局数、已有授权和停止条件。
若工具不能按角色名加载配置，以不继承完整对话的方式显式传入该模型、推理强度与角色指令，
不沿用父任务模型；已有实例设置不符时新建符合设置的 subagent。
负责冻结模型比赛、逐局封存、独立重算 Top 1／Top 4／平均名次与哈希核验；不启动或恢复训练。
默认批量比赛全量并行：指定 N 局就同时调度 N 局，320 局即 320 局一起推进，
每轮跨阵容与环境分片将同模型的 ready 席位合并为 GPU batch，再分发动作推进下一轮。
不能因阵容有十组就串行跑十批 32 局；确有资源限制须报告实测证据和实际并发，优先只切推理 microbatch。
固定总局数不得通过补入额外新局维持满载，不按输赢丢局或重跑。
下一次执行前使用或补齐满足该调度的入口；不追溯改写已封存 32 并行比赛，不虚称配置即实现。
完整协议、调用和验收见 `docs/EVALUATION.md`；角色文件只定义职责，不代表已启动比赛。
新测评还需按玩家/回合记录动作数、可见状态返回、连续返回峰值及强制 End，
汇总每局/每回合动作数均值和 P50/P90/P95/P99、无重复玩家局比例；只读诊断不改变对局规则。
历史只有总动作数时不得声称能还原重复、每回合统计或未记录的 V0 行为，缺失明确标为未知。

## 提交与 Pull Request

提交标题沿用 `Add`、`Fix`、`Defer` 等简短祈使句，不强制 Conventional Commits。
PR 说明问题、行为变化及验证命令和结果；有相关 issue 时关联，实验改动附配置与指标链接，界面改动附截图。
检查 Git diff，确认模型、凭据和临时产物均被忽略；将目录迁移与实际删除区分清楚。
