# 客户端数据源：Power.log 与 HearthMirror

2026-10-31 的选择是保留 Power.log / hslog 作为事件、实体标签和合法动作的主来源，
使用已安装 HSTracker 的 HearthMirror 补齐客户端状态。模型、权重、训练引擎不变。
战场、酒馆位置、发现／英雄选项统一使用 Power.log，不进行实时双源核对或位置覆盖。

## 成熟项目的实际分工

本次核对的 HDT 源码为 `7c0550fe8388bf45e0f3dbc5d2ff25ada2e1a3ae`：

- [LogWatcherManager.cs](https://github.com/HearthSim/Hearthstone-Deck-Tracker/blob/7c0550fe8388bf45e0f3dbc5d2ff25ada2e1a3ae/Hearthstone%20Deck%20Tracker/LogReader/LogWatcherManager.cs)
  仍订阅 Power 日志，分别处理实体／标签、选择与 GameState 包。
- [Watchers.cs](https://github.com/HearthSim/Hearthstone-Deck-Tracker/blob/7c0550fe8388bf45e0f3dbc5d2ff25ada2e1a3ae/Hearthstone%20Deck%20Tracker/Hearthstone/Watchers.cs)
  通过 HearthMirror 的 provider 读取 Scene、Choices、Mulligan、PlayZone 等 UI 状态，
  `GetBoardState(NeedsFriendlyZone)` 返回可见区域信息。招募时 opposing play zone 是酒馆。
- [BoardStateWatcher.Battlegrounds.cs](https://github.com/HearthSim/Hearthstone-Deck-Tracker/blob/7c0550fe8388bf45e0f3dbc5d2ff25ada2e1a3ae/Hearthstone%20Deck%20Tracker/Live/BoardStateWatcher.Battlegrounds.cs)
  组合实体标签、附魔和特殊商店状态，并没有用一个内存快照替代整个事件模型。

HDT 的日志解析器是 C# 实现；AutoTavern 当前使用独立的
[python-hslog](https://github.com/HearthSim/python-hslog)。HearthMirror 不包含这份 Python
依赖。

## 数据来源

| 数据 | 本项目采用的来源 | 原因 |
| --- | --- | --- |
| 本局五种族 | HearthMirror | 日志中缺少完整名单；开局自动读取、按局绑定 |
| 本人玩家编号 | HearthMirror，失败时回退 hslog | 当前客户端直接确认身份，消除仅靠揭牌推断的歧义 |
| 当前界面／切换／旁观状态 | HearthMirror，失败时回退日志记录 | 区分当前 UI 与日志中最后一次界面记录 |
| 战场和酒馆卡牌位置 | Power.log / hslog | 与实体标签、合法动作处于同一事件流，避免混用不同时间点的位置 |
| 发现与英雄选项 | Power.log 的 Choices 包 | 统一保留实体编号、顺序、选择类型与合法目标 |
| 金币、血量、费用、附魔、成长计数 | Power.log 实体标签及公开事件历史 | 当前验证的 Mac 公共接口不提供完整的模型观测 |
| 买卖、施法、选择、目标、执行确认、回放 | Power.log / hslog | 保留有序事件；轮询当前画面无法保证留住两次采样间的所有操作 |
| 模型合法动作与特征编码 | 本项目适配层 | 仍需将真实客户端协议转为训练模型的固定输入与动作空间 |

这份分工基于本机 HSTracker 3.7.2 / 3990 的已验证接口。日志保存有序事件和完整实体
标签，辅助读取补充日志缺少的元数据。模型观测与动作编码继续由本项目适配层完成。

## 接入试验的实测依据

原始记录：`runs/client-observation/20261031T034859Z/`。

- 本局游戏句柄 16304，日志会话 `Hearthstone_2026_10_31_11_44_13`。
- 种族：野兽、元素、机械、鱼人、野猪人。来源为 `hearthmirror`；首次空值后自动重试，
  约 1.16 秒完成两次相同结果确认，随即生成选英雄建议。没有人工填写或 OCR。
- 内存读到 local player 3，与本局日志一致。直接读取场景为 GAMEPLAY。
- 截至 12:13:40，279 次有新增字节的日志增量处理：中位 2.397 ms，P95 10.542 ms；
  1910 次内存采样：中位 3.705 ms，P95 5.245 ms。它们覆盖的数据和采样时间不同，
  不能当作同工作量的 A/B 基准；只说明全面替换日志没有已证实的速度优势。

上述毫秒数是单次数据获取／处理耗时，不是从鼠标操作到建议显示的完整延迟。完整延迟
还包括游戏自身发布数据、后台轮询、包结束、模型推理和窗口刷新。

## 本项目实现

`tools/client_memory.py` 复用独立 native helper 的有界 JSON 通道。后台每 100 ms 读取
界面和本人身份字段，单次 2 秒超时，返回数据超过 750 ms 不再使用。API/UI 线程不等待读取。
异常时仅重建本项目的读取辅助进程，不操作游戏。

内存结果必须属于当前日志会话和 game_key。修改玩家归属前，额外要求匹配种族读取
已经确认的进程／会话／游戏句柄，玩家存在于当前实体表，并连续两次确认身份。
跨局迟到结果、未知身份、旁观和不匹配句柄不会覆盖当前玩家。修改玩家编号会使旧模型
建议失效；仅增加来源确认不会触发重复推理。

场景确认离开对局、界面切换或处于旁观时撤下建议。战场／酒馆位置和发现／英雄选项
仅由 Power.log 产生；native helper 不调用 `getBoardState:`、`getCardChoices` 或
`getMulliganState`。即使旧版 helper 返回卡牌列表，Python 边界也会丢弃这些字段。
没有卡牌差异诊断、双源等待或内存位置覆盖；日志自身的包完整性和合法动作检查保留。

API 的 `client_scene_source`、`perspective.source` 和 `memory_reader` 标明界面、身份的来源与连接状态；
`sessions/<session>/memory.jsonl` 保存脱敏后的状态变化。未知属性、账户、连接凭据、
牌库、对手手牌均不输出。辅助进程签名与构建要求见 [CLIENT_RACES.md](CLIENT_RACES.md)。

回归覆盖跨局隔离、超时、陈旧数据、身份确认、旁观和界面切换，以及旧版 helper
返回的卡牌字段不能修改位置、选项、操作就绪状态或干扰身份确认。测试命令见
[CLIENT_RACES.md](CLIENT_RACES.md)。

12:17 的正式版本完成编译、签名、规范启动和窗口检查。公开状态仅返回 `scene`、
`local_player_id`、`spectator`，u20 建议正常显示。这是阶段性接入验收，不代表整局
无故障或模型强度验证。试验汇总保存在本地 `component-comparison.json`，原始日志、
模型文件和录制数据不进入版本库。
