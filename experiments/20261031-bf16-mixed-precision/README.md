# BF16 / FP32 混合精度对比

日期：2026-10-31，Asia/Shanghai。状态：完成。正式训练保持停止；没有保存测试更新后的模型，也没有修改已有检查点或正式训练配置。

## 精度口径

基线是 **FP32 存储、允许 TF32 计算**，不是严格 IEEE FP32：
`torch.set_float32_matmul_precision("high")`，已核对 `torch.backends.cuda.matmul.allow_tf32 == True`。
该设置不改变权重、梯度、输出的 dtype；具体每个矩阵乘法采用的 kernel 没有逐一 profile，不能把全部运算称为 TF32。
若要禁止 CUDA FP32 矩阵乘法使用 TF32，应改用 `highest`；本次没有测该模式。
依据：[本地开关核验](verification.json)、[PyTorch 精度设置说明](https://docs.pytorch.org/docs/2.15/generated/torch.set_float32_matmul_precision.html)。

混合精度组仅对网络前向启用 `torch.autocast("cuda", dtype=torch.bfloat16)`，
输出 logits、value、placement 转回 FP32 后，再计算动作分布、log probability、PPO ratio、KL 和各项损失。
参数、累积梯度及 Adam 两个 moment 仍为 FP32；反向在 autocast 上下文外执行。
没有将整个模型转为 BF16，没有启用 GradScaler。
实现参考 [PyTorch AMP 文档](https://docs.pytorch.org/docs/2.15/amp.html)。

## 设备与方法

- RTX 5080，16 GB；Python 3.12.8、PyTorch 2.15.0+cu132。
- d=192、4 层、8 个注意力头、FFN=1024，3,491,978 个可训练参数。
- 使用正式实验第三轮后的不可变检查点，同一份参数、Adam 状态和固定 8,192 条真实样本。
- 采样种子 900101，256 个环境、4 个 native 线程、每回合动作上限 60；收集 32,768 条决策后抽取固定 batch。
- 有效 batch 固定 8,192、epoch=1；每次计时更新前恢复相同权重和 Adam。前向、反向、梯度裁剪、Adam 均包含在完整更新中。
- 每档 3 次预热、9 次计时；清理 allocator 缓存后再额外预热一次。逆序补测用于核对顺序与计时波动。
- batch 已驻留 GPU；不包含模拟器采样、GAE、CPU 解包和 H2D。显存是测试进程 PyTorch 峰值 allocated，不是整卡总占用。

## 训练延迟与显存

主表统一采用 [逆序补测](reverse-repeat/results.json) 的 **同步墙钟中位数**。
[初测](results.json) 同样完整保留，没有把两次的最低数值拼成一张表。

| microbatch | 梯度累积 | 基线更新 ms | BF16 更新 ms | 加速倍数 | 基线峰值 GiB | BF16 峰值 GiB | 显存减少 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 32 | 657.3 | 500.7 | 1.31× | 1.45 | 1.09 | 25.1% |
| 512 | 16 | 634.2 | 450.5 | 1.41× | 2.68 | 1.97 | 26.2% |
| 1,024 | 8 | 663.3 | 415.4 | 1.60× | 5.12 | 3.74 | 27.1% |
| 2,048 | 4 | 632.8 | 442.0 | 1.43× | 10.04 | 7.26 | 27.7% |

上述每次更新都消费 8,192 条样本、执行一次 Adam step。256 的两次墙钟中位数范围为
基线 570–657 ms、BF16 439–501 ms；各档原始范围见 [summary.json](summary.json)。
两次测量都支持 BF16 比允许 TF32 的基线更快、且显存更低的结论，不能把单次小幅排名视为稳定最优。
512、1,024 均是可考虑的工作点；2,048 在本次测试里没有带来额外吞吐收益。

CUDA event 区间也保存在原始记录中，主表没有混用它和墙钟。
补测各档 GPU event 更新中位数分别是基线 623/612/612/617 ms、BF16 480/416/413/408 ms。
本次 WSL 环境中它与墙钟存在差异，约 1.3–1.5× 的 GPU 区间加速与墙钟测量方向一致。
GPU event 区间包含主机提交间隙，不能解释为全部 kernel 纯运行时间之和。

## 采样、数据与模型状态

256 条观测的 no-grad 策略前向，补测墙钟中位数从 **6.39 ms 降为 5.77 ms**，约 1.11×。
这只是策略推理；模拟器和 CPU 编码没有因此加速。

CPU rollout 存储仍采用无损压缩格式，完整一轮 4,194,304 条样本约 **13.07 GiB**，autocast 不改变它。
FP32 参数、梯度、Adam 一阶及二阶状态仍约为 `16N` 字节，即 **53.3 MiB**；
BF16 主要减少部分激活的存储，因此总峰值没有减半。检查点精度及保存格式也没有改变。

将三轮既有完整训练的平均耗时（扣除 profiling 暂停）按本次相对速度粗略缩放：

| 配置 | 每轮采样估计 | 每轮更新估计 | 每轮合计估计 | 完整 300 轮估计 |
| --- | ---: | ---: | ---: | ---: |
| 既有基线，micro=256 | 249 s | 370 s | 10.32 min | 51.6 h |
| BF16，micro=256 | 239 s | 282 s | 8.68 min | 43.4 h |
| BF16，micro=512 | 239 s | 253 s | 8.21 min | 41.1 h |
| BF16，micro=1,024 | 239 s | 234 s | 7.88 min | 39.4 h |
| BF16，micro=2,048 | 239 s | 249 s | 8.13 min | 40.7 h |

这些是外推，不是完整 BF16 PPO 轮的实测，不含评估和检查点保存；CPU 解包、H2D、游戏长度变化会影响实际耗时。
仍按 **300 × 4,194,304 = 1,258,291,200** 个决策计算，没有更改成论文 TA-23 的每轮约 105 万决策。
估算公式与输入保存在 summary.json。

## 数值检查

对同一份权重检查 8,192 条样本，并在其中 256 条上比较梯度：

| 指标 | 结果 |
| --- | ---: |
| 采样动作 log-prob 平均绝对差 | 0.000510 |
| FP32→BF16 策略分布 KL 均值 | 2.13 × 10⁻⁷ |
| value 平均绝对差 | 0.00195 |
| 贪心动作发生变化的比例 | 1.31% |
| 梯度相对 L2 差 | 8.88% |
| 梯度余弦相似度 | 0.9962 |
| 梯度有限 | 是 |
| BF16 eval/train 同权重 log-prob 最大差 | 0 |

分布差较小，但 BF16 不是数值完全等价的替换。没有运行长时间 BF16 训练，不能由这些检查推断收敛速度、最终名次或胜率相同。
若以后正式切换，采样与训练网络前向应使用相同精度口径，并继续让概率、ratio、KL、损失与优化器保持 FP32。

## 复现与产物

先确认正式训练已停止，使用新输出目录：

```bash
uv run --locked --group train --extra test \
  python experiments/20261031-bf16-mixed-precision/benchmark.py \
  --output-dir experiments/20261031-bf16-mixed-precision/recheck \
  --sizes 2048 1024 512 256 --modes bf16 fp32 --warmup 3 --repeats 9
```

该脚本只更新独立内存副本；不调用正式训练入口，不暂停或恢复其他进程，不保存新权重，拒绝覆盖已有 results.json。
原始记录内部的 `precision="fp32"` 应按本文基线定义解读。

源模型位于 `runs/20261031-ppo-small-paper300/checkpoints/iteration-0003-before-budget-change.zip`，
SHA-256：`68a136df9af883412e4c9e73f79025cd1835b5879d895056a28b1cf7479d6048`。
测试前后哈希相同；最终确认正式训练进程不存在，实验状态仍为 `stopped_by_user`。

配置及索引见 [experiment.json](experiment.json)，源码哈希见 [provenance.json](provenance.json)，
停止状态和精度开关核验见 [verification.json](verification.json)。模型继续保存在 runs，实验结果保存在 experiments。
