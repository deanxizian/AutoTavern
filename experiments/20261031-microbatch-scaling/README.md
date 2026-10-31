# microbatch 延迟与显存实测

- 日期：2026-10-31（Asia/Shanghai）。状态：完成。
- 设备：NVIDIA GeForce RTX 5080，16 GB。
- 模型：d=192、4 层、3,491,978 个可训练参数，采用正式训练第一轮后的模型快照。
- 有效 batch 固定为 8,192，只改变 microbatch 和梯度累积次数。
- 结论：单个 microbatch 的前后向延迟大致线性增长；固定有效 batch 后，更新耗时基本持平。

## 结果

主结果采用反向顺序补测、充分预热后的 **9 次测量中位数**。
原始结果见 [steady-repeat/results.json](steady-repeat/results.json)，
汇总见 [summary.json](summary.json)。

| microbatch | 累积次数 | 训练前向 ms | 损失＋反向 ms | 前向＋反向 ms | 一次有效 batch 更新 ms | PyTorch 峰值分配 GiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 32 | 7.10 | 13.57 | 20.66 | 638.71 | 1.45 |
| 512 | 16 | 11.69 | 24.76 | 36.44 | 615.13 | 2.68 |
| 1,024 | 8 | 23.77 | 50.64 | 74.55 | 621.31 | 5.13 |
| 2,048 | 4 | 50.61 | 101.87 | 152.36 | 617.57 | 10.04 |

从 256 到 2,048，样本数增加 8 倍，单个 microbatch 的前后向约增加 7.37 倍。
从 512 到 1,024、再到 2,048，前后向分别约增加 2.05、2.04 倍。
但每次有效更新所需的 microbatch 次数相应减少，因此 8,192 条样本的更新均约 0.62～0.64 秒。
512 在本次中位数上比 256 快约 3.7%，差距较小，不能视为稳定显著的吞吐收益。
测试没有支持“增加到 1,024 或 2,048 会大幅加速”的推测。

这些是训练模式（含梯度图）的前向／反向，不是采样时的 no-grad 推理延迟。
GPU 事件区间包含主机提交 kernel 的间隙，不是 kernel 纯运行时间之和。
有效 batch 已驻留 GPU，不含环境采样、CPU 解包或 H2D，因此不能直接替代端到端训练耗时。

峰值显存是该测试进程的 PyTorch `max_memory_allocated`，包含权重、优化器、固定 batch 和激活，
不含桌面、CUDA 上下文及暂停的正式训练进程。PyTorch reserved 峰值分别为
1.70、3.10、5.82、11.45 GiB；这些也不等于 nvidia-smi 的整卡总占用。

## 方法与隔离

1. 校验正式训练 PID 与进程组，仅对该进程组 SIGSTOP，保留模型、环境及采样的内存状态。
2. 从相同模型快照，在独立种子 800001、256 环境中采样 32,768 个决策，随机取固定的 8,192 条。
3. 所有方案使用相同 GPU-resident 数据、模型权重及 Adam 状态；每次有效更新计时前恢复参数和优化器。
4. float32、matmul precision=high、无 AMP；包含策略、价值与名次辅助损失。
5. 用 CUDA events 测前向和损失／反向区间，用同步墙钟测完整有效 batch 更新。
6. 测量结束通过 finally 发送 SIGCONT；已确认正式训练恢复运行，正式 microbatch 仍为 256。

初测为 256→512→1024→2048，每档 2 次预热、5 次计时。
初测 2048 抖动较大，清空 allocator 缓存后的首次分配影响了结果。
因此补测改为逆序 2048→1024→512→256，每档 3 次预热，清空缓存后再额外预热一次，计时 9 次。
初测仍保存在 [results.json](results.json)，未覆盖或丢弃；主结论采用稳态补测。

两次共暂停正式训练约 **72.04 秒**，正式训练配置、权重文件及采样预算均未修改。
暂停区间记录于 `experiments/20261031-ppo-small-paper400/profiling-pauses.jsonl`，
所在正式迭代的原始墙钟耗时包含这段等待，分析纯训练吞吐时应扣除。

## 模型与复现

原始模型快照已归档为
[source_model.zip](../../runs/20261031-microbatch-scaling/source_model.zip)，
SHA-256 为 `7cecc0ea7fe37101b920deef63afc764286feb606cd8bf4dc225396e36736cb4`。
测试不保存更新后的测试权重，也不改正式模型。

确认指定 PID 是当前训练进程，再选一个新的输出目录执行：

```bash
uv run --locked --group train --extra test \
  python experiments/20261031-microbatch-scaling/benchmark.py \
  --training-pid 330674 \
  --model runs/20261031-microbatch-scaling/source_model.zip \
  --sizes 2048 1024 512 256 --warmup 3 --repeats 9 \
  --output-dir experiments/20261031-microbatch-scaling/recheck
```

PID 可能变化，应以正式实验的 `process.json` 为准。入口会验证进程命令，
拒绝接管已暂停的进程，并拒绝覆盖已有结果。
