# PPO 库安装与验证

- 日期：2026-10-31（Asia/Shanghai）。
- 状态：完成。
- 目的：使用 uv 和 Python 3.12 安装后续 PPO 开发环境，验证 GPU、动作掩码、模型保存与加载，以及现有 TavernGym Python 接口。
- 环境：项目根目录 `.venv`，uv 0.9.16，Python 3.12.8，WSL2。

## 已安装版本

| 包 | 版本 |
| --- | --- |
| PyTorch | 2.15.0，CUDA 运行时 13.2 |
| Stable-Baselines3 | 2.9.2 |
| sb3-contrib（MaskablePPO） | 2.9.2 |
| Gymnasium | 1.4.0 |
| TensorBoard | 2.21.0 |
| TavernGym | 1.0.0，本地源码 editable 安装，release 绑定 |
| NumPy | 2.5.4 |
| pytest | 9.2.0 |

依赖声明位于根目录 `pyproject.toml` 的 `train` 组，精确版本保存在 `uv.lock`，
Python 次版本由 `.python-version` 指定。完整依赖见 [packages.txt](packages.txt)。

## 验证结果

1. uv 依赖检查通过，已安装的 49 个包兼容。
2. NVIDIA GeForce RTX 5080（16 GB），计算能力 12.0；PyTorch 包含 `sm_120`，
   CUDA 矩阵乘法和反向传播均通过。驱动版本为 616.56。
3. MaskablePPO 在库自带的合成环境 `InvalidActionEnvDiscrete` 中运行 32 步，
   保存原生 `.zip` 检查点后重新加载，确定性预测一致，另检查 32 次采样动作均合法。
   该检查使用 CPU；临时模型已删除。
4. TavernGym 绑定来自当前引擎源码，源码摘要为 `c807e9aa1a6746ee`。
   245 个动作、3,227 个数值特征和 124 个卡牌索引均可读取。
5. 现有 `test_smoke.py`、`test_env.py`、`test_batch.py` 共 **20 项测试通过**，
   涵盖完整机器人对局、单席位环境、批量环境及保存恢复。

这是安装验证，尚未训练游戏 PPO 模型。游戏训练前仍需接入 SB3 环境包装和动作掩码。

## 复现命令

在项目根目录、已安装所需 Rust 工具链的环境中执行：

```bash
uv sync --locked --python 3.12 --group train --extra test
uv pip check --python .venv/bin/python
uv run --locked --group train --extra test \
  python experiments/20261031-ppo-library-setup/verify.py --require-cuda
uv run --locked --group train --extra test \
  pytest tests/test_smoke.py tests/test_env.py tests/test_batch.py -q
```

`--require-cuda` 要求当前进程能访问 NVIDIA GPU。CPU 机器可省略该选项；
若写入默认输出文件，会替换本目录的验证结果，保留本次结果时请指定 `--output`。
本次会话使用 `/tmp` 中的 Rust 工具链及 uv 缓存，具体命令见
[环境说明](../../docs/TRAINING.md)。

## 产物与后续存储约定

- [verification.json](verification.json)：版本、CUDA、模型存取与源码一致性检查结果，以及依赖文件 SHA-256。
- [verification.txt](verification.txt)：验证脚本原始输出。
- [tests.txt](tests.txt)：20 项接口测试原始输出。
- [sync.txt](sync.txt)、[dependency-check.txt](dependency-check.txt)：依赖同步及兼容性检查输出。
- [verify.py](verify.py)：可重新运行的安装验证脚本。

正式训练的权重与检查点统一放在 `runs/yyyymmdd-{exp-name}/`；
实验配置、说明、TensorBoard 日志、评估指标与模型路径放在同名的
`experiments/yyyymmdd-{exp-name}/`。模型使用用户确认的 SB3 原生 `.zip` 格式。
完整目录约定见 [训练环境与产物说明](../../docs/TRAINING.md)。
