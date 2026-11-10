# 实验 01：PPO 对脚本对手

状态：**完成**。这是第一次正式模型实验；环境准备与小预算试跑归入 [infra](../20261031-infra/README.md)。

| 记录 | 内容 |
| --- | --- |
| [训练与整局评估](20261031-ppo-small-100ppo-32minibatches/README.md) | 100 次 PPO、3,200 次 Adam、26,214,400 条样本，约 57 分钟 |
| [TavernBench](20261031-ppo-small-tavernbench/README.md) | 同一个 best 模型的 368 题局部决策评测 |
| [动作循环诊断](20261031-action-loop-diagnostic/README.md) | 128 局确认换位回退和冻结往返；保存原生可重放案例，未改策略 |
| [可选循环筛选对照](20261031-action-loop-filter/README.md) | 旧权重加历史筛选，128 局受测循环为零；确定性名次改善，采样模式有回退 |
| [BF16 整局对战对照](20261031-bf16-match-evaluation/README.md) | FP32／BF16 × 随机／贪心，各 256 局；配对差值与置信区间 |

模型为 d=192、4 层、约 350 万可训练参数；FP32，允许 TF32；microbatch=1,024，
有效 minibatch=8,192，epoch=1，LR 3e-4→3e-6 线性衰减。对手为七个同策略脚本机器人。

| 指标 | 结果 |
| --- | ---: |
| 独立 128 局平均名次 | 1.765625 |
| 前四率／第一名率 | 93.75% / 58.59% |
| TavernBench 纯策略 Play | 20.96% |
| TavernBench 加 combat256 Play | 43.88% |

最终只保留第 100 轮最佳模型：
[20261031-131146-best_model.zip](../../runs/20261031-ppo-small-100ppo-32minibatches/20261031-131146-best_model.zip)。
权重位于 runs，未上传 Git。配置、指标、原始答案、评分和留存摘要位于上述两个记录目录。

下一次正式模型训练使用 [exp02](../20261031-exp02-mixed-opponents/README.md)。
