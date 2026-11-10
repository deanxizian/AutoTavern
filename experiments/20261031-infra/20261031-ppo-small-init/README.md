# 小模型初始化：d=192，4 层

- 日期：2026-10-31（Asia/Shanghai）。
- 状态：初始化与验证完成，本记录对应零步模型。后续正式训练见
  [400 轮实验](../20261031-ppo-small-paper300/README.md)。
- 模型：[initial_model.zip](../../../runs/20261031-ppo-small-init/initial_model.zip)。
- 实际配置：[config.json](config.json)。
- 初始化元数据：[initialization.json](initialization.json)。
- 模型训练步数 **0**，优化器更新次数 **0**，初始化随机种子 **42**。

## 模型

| 项目 | 当前设置 |
| --- | --- |
| 可训练参数 | **3,491,978**，约 349 万 |
| Transformer | d=192，4 层，8 个注意力头，FFN=1024，Pre-LN，GELU，dropout=0 |
| 输入 | 当前观测组织为 49 个实体 token，带存在性掩码 |
| 卡牌词表 | 3,984 项，0 为空位；静态特征每项 66 维 |
| 卡牌表示 | 身份 embedding＋静态特征投影＋冻结规则文本向量的投影 |
| 输出 | 245 个结构化动作分数、状态价值、8 类名次辅助预测 |
| 模型格式 | SB3 原生 `.zip`，含网络参数、文本向量、算法配置及优化器状态 |
| 初始化设备 | NVIDIA GeForce RTX 5080；存档也可在 CPU 加载 |

参数量包括身份表、各投影、Transformer 和三个头；不计入离线文本编码器，
冻结文本表与静态表作为 buffer 随模型保存。
Transformer 为 2,174,080 参数，身份表为 764,928 参数，其余为投影、汇聚及输出等。

规则文本采用我们选定的
[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，
固定版本 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。
清理英文规则中的 HTML 标签及多余空白，对唯一非空文本编码，缓存 384 维归一化向量；
空文本和 padding 对应零向量。3,983 条卡牌记录中有 3,755 条非空文本，
去重后为 3,462 条。缓存路径为 `runs/assets/card-text/minilm-l6-v2.pt`，
但加载初始化模型不依赖该外部文件或文本编码器。

## 相对论文的实现选择

用户指定采用论文小规模 `d=192、4 层、约 350 万参数`。论文未公开文本编码器、
FFN、归一化位置及指针头的完整实现，这些设置是本地实现选择，
不能声称还原了原始 350 万参数网络的全部细节。

当前 49 个 token 为：全局 1、己方随从 7、手牌 10、商店 8、待选项 16、对手 7。
英雄、技能、饰品、任务等按角色编码后汇聚到全局 token；对手上次可见的随从
经过位置编码和非线性汇聚后进入对应对手 token。
动态属性使用已由模拟器缩放的数值并做 symlog；卡牌 ID 始终为类别索引。
动作评分读取全局、来源实体、目标实体或插入位置，以及动作类型表示。

输入仅来自现有玩家视角观测。公开观测没有列出全部附件 ID、持久效果细节和
每个历史对手的战斗结果，当前只编码已提供的字段（例如附件数量）。
完整复现论文输入还需补充这些玩家可见字段。

## 训练就绪设置

已实现 Gymnasium 包装、SB3 策略与 `EntityPPO`，可以从初始化存档加载训练。
名次辅助监督仅用于当前 rollout 中已得到最终名次的片段，跨 rollout 的未完成片段
不伪造标签。辅助标签不会进入观测。首次从零训练没有前一阶段 KL 锚定。

后续默认配置：8 个环境 × 256 步，batch size 256、1 epoch、学习率 `1e-4`；
gamma=1、lambda=0.995、clip=0.2、target KL=0.05、梯度裁剪=1；
熵／价值／辅助系数为 0.002／0.5／0.1。
采样规模和初始学习率是本地起始配置，其余若干系数参考论文表 7，
整套设置不能视作论文早期小模型训练配方。

对手为七个共用相同策略的内置脚本机器人。奖励沿用模拟器的线性名次效用：
第 1 名 +1，第 8 名 −1，中间线性插值；它不等于论文后期以第一名为目标的完整训练条件。
论文最终阶段的校准对手库及多阶段历史均未复现。

## 验证

- 新增 6 项测试通过，覆盖动态属性编码、无卡牌 ID 的有效选项、padding 隔离、
  非法动作零概率、三个头及文本投影的梯度、辅助监督边界、SB3 存取和继续训练。
- 与现有环境测试一起运行时共 26 项通过；测试中的训练仅使用极小合成环境与临时模型。
- 实际存档在 CPU 与 CUDA 新进程中加载，各自检查 32 次真实模拟器合法动作。
- 实际网络的三个头均能反向传播，梯度有限且非零；只做梯度检查，没有优化器更新。
- 保存加载后权重逐项完全一致，输出在 `rtol=atol=1e-6` 内一致。
- 上述检查前后初始化文件 SHA-256 不变；始终保持零训练步数。

原始输出与结果：
[initialization.txt](initialization.txt)、[tests-learning.txt](tests-learning.txt)、
[verification-cpu.json](verification-cpu.json)、[verification-cuda.json](verification-cuda.json)。
模型及源文件摘要另存于 [artifacts.json](artifacts.json)。

## 命令

本次初始化命令（输出文件已存在，入口会拒绝覆盖）：

```bash
uv run --locked --group train --extra test python -m taverngym.learning.initialize \
  --config configs/ppo-small.json \
  --run-dir runs/20261031-ppo-small-init --device cuda
```

后续开始训练时执行以下命令；**本次未执行**：

```bash
uv run --locked --group train --extra test python -m taverngym.learning.train \
  --model runs/20261031-ppo-small-init/initial_model.zip \
  --run-dir runs/20261031-ppo-small-baseline \
  --total-timesteps 1000000 --device cuda
```

训练产物写入新的 `runs/` 目录，保留初始化文件。开始正式实验时，仍需按
`experiments/yyyymmdd-{exp-name}/` 约定记录评估方案、原始输出和结果。
如从新环境重建文本缓存，使用以下命令；缓存输出路径需不存在：

```bash
uv run --locked --group train --extra test python -m taverngym.learning.text \
  --output runs/assets/card-text/minilm-l6-v2.pt \
  --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --device cuda
```

复查初始化模型而不更新其权重：

```bash
uv run --locked --group train --extra test \
  python experiments/20261031-infra/20261031-ppo-small-init/verify.py --device cuda
```
