# 第一版：100 个 minibatch

日期：2026-10-31，Asia/Shanghai。状态：完成，100 次 Adam 更新、819,200 条样本已核验。
按用户要求，评估后本次 runs 目录只保留 best_model.zip，其余七份模型已清理；指标和 TensorBoard 均保留。

按用户要求从已保存的零步初始化模型开始，训练 **100 个有效 minibatch，即 100 次 Adam 更新**。
本次采用小预算配置，每次采集 8,192 条新样本，然后更新一个 minibatch；不是 100 个 microbatch，
也不是原 420 万样本／轮的 100 轮训练。旧 300 轮实验保持停止，不从其三轮模型继续。

## 配置

| 项目 | 设置 |
| --- | --- |
| 模型 | d=192、4 层、8 头、FFN=1024，3,491,978 个可训练参数 |
| 起点 | `runs/20261031-ppo-small-init/initial_model.zip`，训练步数与 Adam 状态均为零 |
| 采样 | 32 个原生环境 × 每环境 256 步 = 8,192 条／轮 |
| 有效 minibatch | 8,192，epoch=1，每轮一个 |
| microbatch / 累积 | 1,024 / 8 |
| 总预算 | 100 轮 × 一个 minibatch = 100 次 Adam 更新，819,200 条样本，800 次 microbatch 反向 |
| LR | 初始 `3e-4`，线性衰减至 `3e-6`，无 warmup、无重启 |
| 精度 | FP32 参数、梯度和 Adam；允许 TF32 矩阵乘法；没有 BF16 autocast |
| γ / λ | 1.0 / 0.995 |
| clip / target KL / grad clip | 0.2 / 0.05 / 1.0 |
| 熵 / value / 名次辅助系数 | 0.002 / 0.5 / 0.1 |
| 对手／动作预算 | 七个相同脚本策略；每回合 60 次操作上限 |
| 评估 | 初始 64 局；每 25 轮 64 局；最终独立种子 128 局；随机流与训练隔离 |
| 保存 | 每 25 轮检查点，另有 starting/latest/best/final |

首版的小 rollout 增加了策略刷新频率，与旧 256 环境 × 16,384 步的配置不同。
本次结果不能直接用于对比此前完整轮的样本效率或计算吞吐。

## 学习率时序

沿用 SB3 的时序：采样完本轮，再按全局已采样决策数更新 LR，随后执行 Adam。
因此配置初值为 3e-4，第一次实际 Adam 更新使用 2.9703e-4，最后一次为 3e-6。

| 时点 | LR |
| --- | ---: |
| 初始化，尚未更新 | 3e-4 |
| Adam 第 1 次 | 2.9703e-4 |
| Adam 第 25 次 | 2.2575e-4 |
| Adam 第 50 次 | 1.515e-4 |
| Adam 第 75 次 | 7.725e-5 |
| Adam 第 100 次 | 3e-6 |

本次每轮只有一个 minibatch，日志中的 approx_kl 和 clip_fraction 在该次 Adam step 前计算，
此时采样策略与待更新策略相同，数值接近零不能说明更新后的策略变化同样小。
学习效果以固定协议下的名次评估为准；短预算也不足以判断完整训练的收敛结果。

论文仅公布最终 TA-23 阶段的 6e-6→5e-6，未公布早期从零训练的 LR 配方。
本实验采用用户选定的 3e-4；3e-6 终值沿用本轮讨论的候选建议，均属本地设置。

## 目录与命令

配置、指标、评估、来源摘要和结论保存在本实验目录；模型保存在
`runs/20261031-ppo-small-v1-100-minibatches/`。
配置及预算见 [config.json](config.json)，最终核验见训练完成后生成的 verification.json。

首次执行命令如下；模型目录已存在时不要重复运行，复现应选择新的目录。

```bash
uv run --locked --group train --extra test python -m taverngym.learning.experiment \
  --config experiments/20261031-ppo-small-v1-100-minibatches/config.json \
  --model runs/20261031-ppo-small-init/initial_model.zip \
  --experiment-dir experiments/20261031-ppo-small-v1-100-minibatches \
  --run-dir runs/20261031-ppo-small-v1-100-minibatches --device cuda
```

模型初始 SHA-256 为 `3c8f024a0cfb1526298d0d9a105d4d5a2e283cc724eb0682b160f0a9a7d38e53`，
初始文件和旧 300 轮实验均保留原样。

## 结果与模型留存

训练共耗时约 **221 秒（3 分 41 秒）**，含训练中的评估和模型保存。
最后一次 Adam 更新确实使用 3e-6，全部 100 次更新的学习率符合预设线性曲线。
完整训练模型在清理前已核验 Adam step=100、参数有限及初始模型未被修改；核验结果见
[verification.json](verification.json)，汇总见 [results.json](results.json)。

固定评估种子的 64 局结果：

| 更新次数 | 平均名次 | 前四率 | 吃鸡率 |
| ---: | ---: | ---: | ---: |
| 0 | 7.921875 | 0% | 0% |
| 25 | **7.859375** | 0% | 0% |
| 50 | 8.000000 | 0% | 0% |
| 75 | 7.968750 | 0% | 0% |
| 100 | 7.953125 | 0% | 0% |

最优版本在独立评估前已按以上平均名次选定，为第 **25 次更新**，对应 204,800 条训练样本。
同一组独立种子下的 128 局复核，三个版本均为 0% 前四率和 0% 吃鸡率：

| 版本 | 平均名次 |
| --- | ---: |
| 零步初始化 | 7.9765625 |
| 预选最优，第 25 次更新 | 7.9453125 |
| 完整训练，第 100 次更新 | 7.9609375 |

这点差异不足以说明取得实质提升；当前首版仍基本处于后位，不代表训练收敛。
独立种子没有用于更换最佳版本，避免用测试结果再次选模。

本次预算只有 100 个 minibatch；先前完整 300 轮方案为每轮 512 个 minibatch，
共 153,600 次 Adam 更新、1,258,291,200 条样本。当前试跑的样本量是该方案的 1/1,536。
先前 52–54 小时（BF16 外推 39–42 小时）的估计对应全量方案，不是本次试跑；
本次完成不意味着完成了原 300 轮训练，旧实验仍停止。

只保留 `runs/20261031-ppo-small-v1-100-minibatches/best_model.zip`，约 47.54 MiB。
完整训练完成了 100 次更新，保留权重来自第 25 次更新；两者应区分。
删除七份 starting/latest/final/定期检查点，回收约 305.47 MiB；
文件路径、大小及 SHA-256 均留在 [retention.json](retention.json)。零步初始化和其他实验未清理。
清理脚本为 retain_best.py；verify.py 在清理后会验证保留模型和先前完整训练的核验记录。

## TensorBoard

训练标量和补充导出的评估曲线保存在本实验目录，清理权重不会删除它们。
本机的 TensorBoard 在启动时会卡在对关闭端口的无超时探测，使用以下启动器为探测设置一秒超时，
服务启动后恢复默认 socket 行为；未修改第三方包：

```bash
uv run --locked --group train --extra test \
  python experiments/20261031-ppo-small-v1-100-minibatches/serve_tensorboard.py
```

默认地址为 http://127.0.0.1:6006/ ，Codex 浏览器可能映射到其他本地端口。
可查看 train/learning_rate、各项 loss、entropy、KL、clip fraction，
以及 eval/mean_placement、eval/top4_rate、eval/first_rate 和对应的 heldout 曲线。
评估曲线由 export_eval_tensorboard.py 从已有 JSON 导出，没有重新训练或改变评估结果。

横轴 **Step = 累计受控玩家决策数**，每完成一个 minibatch 记录一次，因此每次增加 8,192。
本实验 `Adam 更新次数 = Step / 8,192`：204,800 为第 25 次，819,200 为第 100 次。
每次 Adam 更新又包含 8 个 microbatch；Step 不是对局数，也不是 microbatch 数。
