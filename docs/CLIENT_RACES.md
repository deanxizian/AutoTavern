# macOS 自动读取本局种族

种族读取通过本机已安装 HSTracker 随附的 HearthMirror 完成，不使用 OCR、截屏或屏幕
录制权限。HSTracker 主程序不需要运行。模型、权重和训练引擎没有变更。

## 数据来源

参考 HDT 的
[GetAvailableRaces / ReadAvailableRacesFromMemory](https://github.com/HearthSim/Hearthstone-Deck-Tracker/blob/7c0550fe8388bf45e0f3dbc5d2ff25ada2e1a3ae/Hearthstone%20Deck%20Tracker/Hearthstone/BattlegroundsUtils.cs)：
通过 HearthMirror 读取游戏的可用种族，数据未初始化时重试，并按对局缓存。

macOS 使用已安装 framework 的 Objective-C 接口 `initWithPID:withBlocking:`、
`getAvailableBattlegroundsRaces`、`getLogSessionDir`、`getGameType` 和
`getGameServerInfo.gameHandle`。接口也与公开的
[macOS 调用示例](https://github.com/an-stu/battlegrounds_companion/blob/c36cb6b6ae02daebcf3f231cef3054c17a22ff93/macos/AppMain.swift)
相符。采用的是现场安装的供应商组件，不引入示例项目的二进制。

种族请求读取种族数字 ID 和用于校验来源的进程、日志会话、对局句柄与游戏类型。
辅助进程另支持公开界面状态请求，分工与验证见 [CLIENT_DATA_SOURCES.md](CLIENT_DATA_SOURCES.md)。
不输出连接凭据、账号、牌库、其他玩家手牌或整个内存对象；framework 的原始 stdout /
stderr 不会写入本地记录，也不会上传数据。

## 组件与启动

- 本地依赖：`/Applications/HSTracker.app` 或 `~/Applications/HSTracker.app`。
  当前实测版本为 3.7.2 / 3990，framework 含 arm64 与 x86_64。
- 加载前通过 macOS Security 校验 framework 的 Apple 签名链、HearthSim 团队
  `RL7C49LAMC` 和标识 `com.ifehervari.HearthMirror`。不复制、修改或分发供应商二进制。
- `script/build_and_run.sh` 编译 `tools/client_mirror/main.mm`，生成独立
  `AutoTavernMirrorReader`，并携带 `com.apple.security.cs.debugger` 声明。
  这是 HSTracker 使用的进程读取权限；没有修改游戏签名、系统授权数据库或 SIP。
- Apple 说明该权限可能触发系统管理员授权窗口，参见
  [Debugging tool entitlement](https://developer.apple.com/documentation/BundleResources/Entitlements/com.apple.security.cs.debugger)。
  遇到拒绝时报告 `memory_access_denied`，不会自动更改系统权限。
- 默认仍通过 `./script/build_and_run.sh --verify` 启动助手。

组件自检（不开始游戏、不改变游戏状态）：

```bash
runs/client-window/AutoTavernAssistant.app/Contents/MacOS/AutoTavernMirrorReader --probe
```

## 时序和故障处理

`tools/client_races.py` 在后台线程与独立辅助进程通信，API/UI 轮询不等待内存读取。
每次请求只绑定一个日志 game_key；跨局返回的旧结果丢弃。校验当前内存日志会话与
Power.log 所属会话相同，对局句柄在读取前后不变，且该进程/会话/句柄没有绑定到另一局。
再连续读取两次相同的五种已知种族才采用。未知种族、重复值、混合新旧数据或不完整
列表均不推测补齐。当前只接入模型支持的单人战棋类型。

空结果从 250 ms 起退避，最长每 2 秒重试；辅助进程超时 8 秒即结束并重建该辅助进程。
游戏结束、模式切换或日志 game_key 变化时丢弃进行中的旧请求。成功后按 game_key
持久化，重启助手可恢复，同时恢复旧句柄的归属以防跨重启串局；新局重新读取。
API 手动填写不能伪装成内存来源。

实际来源字段为 `lobby_races_source: hearthmirror`，状态字段为 `race_reader`。
本地 `runs/client-window/sessions/<session>/races.jsonl` 按状态变化和每十次重试记录
状态、次数、耗时和校验字段，权限为 0600。主窗口不再要求每局先点击五种族，手动入口
折叠在“手动填写种族”内。

## 验证边界

自动测试覆盖空值重试、未知/重复 ID、异变种族、跨局迟到结果、句柄复用、游戏结束、
读取超时、缺失组件、日志脱敏、按局持久化，以及成功结果自动送入模型。
运行相关回归：

```bash
runs/model-env/bin/python -m pytest tests/test_client*.py tests/test_manual*.py -q
./script/build_and_run.sh --verify
```

自动测试使用模拟读取器；实际组件连接另由辅助进程自检和现场对局验证。

本机已验证供应商签名、动态加载、所需接口和当前炉石进程连接。大厅实测读到
`Hearthstone_2026_10_31_11_44_13`，与实际日志会话一致，`memory_access=true`；首次
连接约 83 ms，后续读取约 1 ms。证据在 `runs/client-native/races-validation/lobby-probe.json`。
随后 12:03 开始的对局完成了开局种族验收：自动读到野兽、元素、机械、鱼人、野猪人，
两次确认后以 `hearthmirror` 来源送入模型，选英雄建议自动出现。详见数据源报告。
