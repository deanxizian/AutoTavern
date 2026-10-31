# 基础设施与试跑

这一组统一记录正式实验前的准备、正确性验证和性能测量，不占正式实验编号。

| 记录 | 用途／状态 |
| --- | --- |
| [脚本整局演示](20261031-scripted-bot-game/README.md) | 验证八人游戏可完整运行 |
| [论文吞吐与接口](20261031-paper-reproduction/README.md) | 单核速度、合法动作、观测与存档恢复 |
| [uv / Python / PPO 环境](20261031-ppo-library-setup/README.md) | 依赖、CUDA 和存取验证 |
| [模型初始化](20261031-ppo-small-init/README.md) | d=192、4 层，保存零步模型 |
| [大 rollout 预演](20261031-ppo-small-paper300/README.md) | 完成 3 轮后停止，保留历史记录 |
| [100 次 Adam 试跑](20261031-ppo-small-v1-100-minibatches/README.md) | 验证训练、评估与最佳模型留存链路 |
| [microbatch 性能](20261031-microbatch-scaling/README.md) | 延迟、显存与梯度累积比较 |
| [BF16 / FP32 性能](20261031-bf16-mixed-precision/README.md) | 精度与吞吐比较，未切换正式训练精度 |
| [操作／循环训练奖励](20261031-reward-shaping/README.md) | 将成本接入 PPO 与 W&B，供第二次训练使用 |
| [混合对手 GPU 预跑](20261031-mixed-opponent-preflight/README.md) | 真实 V1 继承、固定对手、阶段重置与 GPU 吞吐验证 |
| [对手数量与 BF16 推理](20261031-opponent-inference-scaling/README.md) | 1／4 个模型对手采样对照，FP32／BF16 推理延迟与概率差异 |
| [BF16 训练入口](20261031-bf16-training-integration/README.md) | 采样／更新／固定 bot 共用 BF16 前向，FP32 训练状态与存取验证 |
| [独立 Actor 采样](20261031-unified-actor-sampling/README.md) | 6 脚本 + 1 固定 V1 + 1 学习者，BF16 初始采样原型提速 5.51 倍 |
| [Actor 对手数量](20261031-actor-opponent-scaling/README.md) | 新调度下 1／2 个固定 V1 对照；2 个对手采样耗时多约 11%，含训练时间外推 |
| [Actor 接入 PPO](20261031-actor-ppo-integration/README.md) | 压缩缓存、GAE、奖励与 W&B 接入；真实首轮完整预检 58.9 秒、99 项测试通过 |
| [批量动作失效修复](20261031-batch-action-deferral/README.md) | 跨席位死亡触发发现时刷新旧决策，103 项测试通过；从第 22 轮继续 |
| [主动 EndTurn 短测](20261031-endturn-behavior/results.json) | PPO 66，2 固定 V1 + 5 脚本；16 局 × 128 步／模式，随机与贪心均观察到仍有其他合法动作时主动 End；非整局比例 |
| [每局步数分位数](20261031-episode-length-quantiles/results.json) | PPO 25／50／75／81 的四个最近 100 局窗口；2 固定 V1 + 5 脚本、学习者随机采样；P50／P90／P95／P99 与原始长度，无新增训练 |

正式模型实验从 [exp01](../20261031-exp01-ppo-scripted/README.md) 开始；原始测量和停止的试跑均保留。
