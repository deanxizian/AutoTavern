# BF16 前向接入训练入口

2026-10-31 用户确认后续采用 BF16 autocast 前向，FP32 概率与训练状态。
另确认待用配比为 **6 脚本 + 1 固定 V1 bot + 1 学习者**。
本次只接入配置并验证，第二轮保持停止，没有改写已有模型或启动正式训练。

```json
"forward_precision": "bf16",
"opponent_forward_precision": "bf16"
```

- `EntityPolicy` 的动作采样、概率重算、bootstrap value 和普通 predict 共用前向路径。
- neural forward 使用 BF16 autocast；logits、value、辅助头输出转回 FP32，概率与 PPO loss 保持 FP32。
- 参数、梯度累积、Adam 与保存的浮点权重均为 FP32，不启用 GradScaler。
- `EntityPPO.set_forward_precision()` 同时更新 SB3 的 policy_kwargs，新保存的模型保留计算模式。
- 初始化和两个训练入口均已接入；正式入口与通用入口支持 `--forward-precision bf16`。
- 旧配置缺失该字段时继承检查点；旧检查点没有该字段时为 FP32，历史模式不会自动变更。
- 固定对手缺省跟随学习者，可通过 `opponent_forward_precision` 单独覆盖；held-out suite 使用相同设置。

待用片段见 [next-settings.json](../../20261031-exp02-mixed-opponents/next-settings.json)，
它需合入未来确定的完整训练配置，不是启动指令。
此前混合采样动作拒绝错误仍需另外定位；同步采样导致的小批次等待也尚未优化。

## 验证

完整 Python 套件 **92 项通过**，包括真实 CUDA 的 BF16 采样／更新重算一致性、
FP32 梯度／Adam／rollout 数组、保存加载、旧模型兼容和 W&B 离线入口配置。
1 个固定 bot 与学习者在多局中同时使用 BF16，只有学习者执行测试优化器更新，固定 bot 权重保持不变。
所有优化器测试均使用临时微型网络，不会训练已有 V1 或第二轮模型。

此外 [verify_gpu.py](verify_gpu.py) 用真实 V1 权重执行 16 局 × 32 个学习者前向步，
验证双方 raw logits 为 BF16、概率／value／log probability 为 FP32，
eval 与 update 模式的重算一致。该验证完全不调用优化器，不产生新模型。
结果见 [gpu-validation.json](gpu-validation.json)，整体检查与源码摘要见 [verification.json](verification.json)。

```bash
.venv/bin/python -m pytest -q
.venv/bin/python experiments/20261031-infra/20261031-bf16-training-integration/verify_gpu.py
```

第一轮模型直接切换推理精度的完整对战结果另见
[配对整局评估](../../20261031-exp01-ppo-scripted/20261031-bf16-match-evaluation/README.md)。
