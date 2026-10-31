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

## 测试约定

Python 测试命名为 `test_*.py`；Rust 测试位于模块内及 `crates/taverngym/tests/`。
未设置覆盖率百分比门槛；规则、合法掩码、终局及采样修复应补充固定种子的回归测试。
基础 CI 可能跳过训练依赖相关测试，完整验证须安装 `train` 组；报告 CUDA 测试是否实际执行。

## 实验、评估与配置

日期按北京时间，目录使用 `yyyymmdd-{exp-name}`，分组遵循 `experiments/README.md`。
最佳模型命名为 `YYYYMMDD-HHMMSS-best_model.zip`。模型权重、密钥和 SDK 缓存不得提交 Git。
续训只使用完整 PPO 轮检查点；保留失败现场，不改写历史配置和来源摘要。

训练默认将指标与配置记录到 W&B 项目 `eric_eric_/AutoTavern`；train 横轴为实际 Adam 步数，起点评估放在 val 的 step=0。
每次汇报和表格须明确对手数量与版本、学习者随机／贪心模式及局数，例如“2 个固定 V1 + 5 个脚本，随机采样，64 局”。
基准推理不得读取标签，封存预测后再评分；使用已有官方评分器，测试集不参与选模。

## 提交与 Pull Request

提交标题沿用 `Add`、`Fix`、`Defer` 等简短祈使句，不强制 Conventional Commits。
PR 说明问题、行为变化及验证命令和结果；有相关 issue 时关联，实验改动附配置与指标链接，界面改动附截图。
检查 Git diff，确认模型、凭据和临时产物均被忽略；将目录迁移与实际删除区分清楚。
