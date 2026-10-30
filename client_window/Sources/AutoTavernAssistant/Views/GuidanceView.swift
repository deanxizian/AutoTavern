import SwiftUI

struct GuidanceView: View {
    @Bindable var store: SessionStore
    @AppStorage("assistant.compact") private var compact = false
    @State private var selectedRaces: Set<String> = []
    private let races = [("BEAST", "野兽"), ("DEMON", "恶魔"), ("DRAGON", "龙"), ("ELEMENTAL", "元素"),
        ("MECHANICAL", "机械"), ("MURLOC", "鱼人"), ("PIRATE", "海盗"), ("QUILBOAR", "野猪人"),
        ("UNDEAD", "亡灵"), ("ABERRATION", "异变"), ("NAGA", "纳迦")]

    private var proposal: Proposal? {
        guard store.connected, let proposal = store.state?.proposal,
              proposal.expiresAt > Date().timeIntervalSince1970 else { return nil }
        return proposal
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            if let state = store.state, !state.isLive || proposal != nil {
                Text(state.stateTitle).font(.system(size: 13, weight: .medium)).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: compact ? 14 : 18) {
                if let proposal {
                    if !compact {
                        Label("当前这一步", systemImage: "arrow.turn.down.right")
                            .font(.system(size: 12, weight: .medium)).foregroundStyle(.tint)
                    }
                    Text(proposal.title).font(.system(size: compact ? 22 : 26, weight: .semibold))
                    Text(proposal.instruction)
                        .font(.system(size: compact ? 16 : 18, weight: .medium)).lineSpacing(compact ? 4 : 6)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("current-instruction")
                    if let cost = proposal.cost, !compact {
                        Label(cost.label, systemImage: "circle.circle")
                            .font(.system(size: 13, weight: .semibold)).foregroundStyle(.secondary)
                    }
                } else {
                    Image(systemName: store.state?.message == nil ? "hourglass" : "checkmark.circle")
                        .font(.system(size: 29, weight: .light)).foregroundStyle(.secondary)
                    Text(emptyTitle).font(.system(size: 25, weight: .semibold))
                    Text(emptyDetail).font(.system(size: 14)).foregroundStyle(.secondary)
                        .lineSpacing(5).fixedSize(horizontal: false, vertical: true)
                }
                if store.state?.isLive != true {
                Divider().padding(.top, 4)
                HStack(spacing: 10) {
                    Button("已执行") {
                        if let proposal { Task { await store.send("feedback", values: ["proposal_id": proposal.id, "outcome": "executed"]) } }
                    }
                    .buttonStyle(.borderedProminent).frame(maxWidth: .infinity)
                    .accessibilityIdentifier("feedback-executed")
                    Button("未执行") {
                        if let proposal { Task { await store.send("feedback", values: ["proposal_id": proposal.id, "outcome": "not_executed"]) } }
                    }
                    .buttonStyle(.bordered).frame(maxWidth: .infinity)
                    .accessibilityIdentifier("feedback-not-executed")
                }
                .controlSize(.large).disabled(proposal == nil || store.busy)
                }
            }
            .padding(compact ? 16 : 22).frame(maxWidth: .infinity, alignment: .leading)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18))
            .overlay(RoundedRectangle(cornerRadius: 18).stroke(.quaternary, lineWidth: 1))

            if let state = store.state {
                if state.needsLobbyRaces == true {
                    VStack(alignment: .leading, spacing: 10) {
                        Button("从游戏画面读取种族") { Task { await store.scanLobbyRaces(requestAccess: true) } }
                        if let message = store.raceScanMessage { Text(message).font(.caption).foregroundStyle(.secondary) }
                        Text("确认本局种族 · 已选 \(selectedRaces.count) / 5").font(.headline)
                        Text("开局确认一次，之后自动跟随局面更新。").font(.caption).foregroundStyle(.secondary)
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 70))], spacing: 8) {
                            ForEach(races, id: \.0) { race in
                                Button {
                                    if selectedRaces.contains(race.0) { selectedRaces.remove(race.0) }
                                    else if selectedRaces.count < 5 { selectedRaces.insert(race.0) }
                                } label: {
                                    Label(race.1, systemImage: selectedRaces.contains(race.0) ? "checkmark.circle.fill" : "circle")
                                        .frame(maxWidth: .infinity)
                                }.tint(selectedRaces.contains(race.0) ? .accentColor : .secondary)
                                    .buttonStyle(.bordered)
                            }
                        }
                        Button("确认种族，开始建议") {
                            Task { await store.send("lobby-races", values: ["game_key": state.gameKey ?? "", "races": selectedRaces.sorted().joined(separator: ",")]) }
                        }.buttonStyle(.borderedProminent).disabled(selectedRaces.count != 5 || store.busy)
                    }
                }
                if state.canNextFrame {
                    Button { Task { await store.send("next-frame") } } label: {
                        Label(state.nextFrameLabel, systemImage: "arrow.right")
                            .frame(maxWidth: .infinity)
                    }
                    .controlSize(.large).disabled(store.busy || !store.connected)
                    .accessibilityIdentifier("next-frame")
                }
                if state.isLive {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(state.modelStatus?.title ?? "尚未选择模型").font(.system(size: 13, weight: .semibold))
                        Text(state.modelStatus?.detail ?? state.stateDetail)
                            .font(.system(size: 12)).foregroundStyle(.secondary).lineSpacing(4)
                    }
                    .padding(.horizontal, 2)
                } else {
                    Text(state.stateDetail).font(.system(size: 12)).foregroundStyle(.secondary).lineSpacing(3)
                }
            }
        }
        .onChange(of: store.state?.gameKey) { selectedRaces = [] }
    }

    private var emptyTitle: String {
        if !store.connected { return store.state == nil ? "正在连接" : "连接已暂停" }
        if store.state?.isLive == true && store.state?.gameStatus == "complete" { return "本局已结束" }
        if store.state?.message != nil { return "等待新的局面" }
        if store.state?.freshness == "stale" { return "上次局面已过期" }
        if store.state?.needsLobbyRaces == true { return "确认本局种族" }
        if store.state?.readStatus == "snapshot" { return store.state?.modelStatus?.title ?? "正在更新建议" }
        if store.state?.readStatus == "incomplete_snapshot" { return "等待局面稳定" }
        if store.state?.readStatus == "read_or_parse_error" { return "日志需要核对" }
        return "等待对局局面"
    }

    private var emptyDetail: String {
        if !store.connected { return "连接恢复前，操作指引会保持暂停。" }
        if store.state?.isLive == true && store.state?.gameStatus == "complete" { return "本局记录已读取。新对局开始后，助手会自动切换到新的局面。" }
        if let message = store.state?.message { return message }
        if store.state?.clientRunning == false { return "打开炉石后，助手会自动检测新的日志会话。" }
        if store.state?.freshness == "stale" { return "当前记录已过期。等待客户端写入新的局面，旧指引不会继续使用。" }
        if store.state?.readStatus == "snapshot" { return store.state?.modelStatus?.detail ?? "已读取日志中的可见实体，请在“局面核对”查看。" }
        if store.state?.readStatus == "incomplete_snapshot" { return "日志仍在更新，等待完整局面后重新核对。" }
        if store.state?.readStatus == "read_or_parse_error" { return "当前日志无法完整解析，请保留当前会话以便检查。" }
        if let scene = store.state?.clientScene { return "最新界面记录：\(scene)。尚未读取到酒馆对局局面。" }
        return "客户端已打开。收到可读取的局面后，会在这里显示核对结果。你可以继续停留在大厅。"
    }
}
