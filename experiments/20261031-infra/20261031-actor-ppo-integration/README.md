# 独立 Actor 接入正式 PPO

2026-10-31。将先前的采样原型接入 `EntityPPO`，正式配置通过 `sampling_driver=unified_actor` 选择。
原同步路径继续可用，验证环境与训练环境分离。

`ActorOpponentVectorEnv` 复用原生 BatchEnv，无须重新编译。采样在各局中独立推进，按策略身份合批；
每次完成一批学习者决策后，将真实公开观测、action、old value、old log-prob 与奖励
批量压缩并写入 `[rollout_step, game]`，核对各局连续序号及精确配额。
按局保留终局信息、奖励累计和动作上限统计；已结束片段取得名次标签，未结束片段不伪造标签。
完成全部配额后，以各局最后一个 done 和 bootstrap value 计算 FP32 GAE，再执行原有梯度累积更新。

W&B 保持每 Adam 的 train 指标、每 PPO 的采样汇总和五个时点的 val。
新增 `system/actor_*` 记录推理批次、调用数及时间。Actor 动作拒绝时仍停止训练并保存原生诊断，
不会丢弃样本、静默替换动作或将固定对手动作混入 learner 缓存。

## 验证

- 新旧调度的确定性逐局轨迹一致，覆盖 1／2 个固定角色、1／3 线程、多次终局及连续 rollout。
- 按不同顺序写入的压缩观测、动作、奖励、episode starts、名次标签、GAE 和 returns 逐位一致。
- BF16 Actor PPO 能跨两轮更新；固定策略不变，奖励／回合上限统计齐全，Adam 与精度可保存恢复。
- CLI 的离线 W&B 测试覆盖两种采样驱动、每次 Adam、step=0、四条件 val 和关闭 TensorBoard。
- 完整 Python 测试 **99 passed，30.25 秒**，含 CUDA；Rust fmt 通过。

原生完整预算测试：256 局 × 1,024 决策，完成 262,144 条和 237 次终局，无动作拒绝。
结果：[native-rollout.json](native-rollout.json)。

| 完整 PPO 预检 | 实测 |
| --- | ---: |
| 实际首轮学习率 | 2.9703e-4 |
| 新样本 | 262,144 |
| 已结束对局 | 243 |
| Adam 更新 | 32 |
| 含压缩缓存和指标的采样 | 41.56 秒 |
| 更新 | 17.25 秒 |
| 两项合计 | 58.81 秒 |
| 峰值 PyTorch allocated | 3.75 GiB |

结果：[full-ppo-preflight.json](full-ppo-preflight.json)，运行脚本：[preflight.py](preflight.py)。
预检更新只作用于一次性内存模型，不保存模型、不开正式 W&B；正式阶段重新从 V1 开始。
另保留第一次以结束学习率 3e-6 执行的 [低学习率预检](low-lr-preflight.json)，正式首轮稳定性以表中复检为准。
此前 4 固定对手尝试中的动作拒绝根因没有被本次无错误预检单独证明或修复，运行时仍严格检查并保存诊断。
