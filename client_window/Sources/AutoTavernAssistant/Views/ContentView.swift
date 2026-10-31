import SwiftUI

struct ContentView: View {
    @Bindable var store: SessionStore
    @AppStorage("assistant.pinned") private var pinned = true
    @AppStorage("assistant.compact") private var compact = false
    @State private var tab = "guidance"

    var body: some View {
        VStack(spacing: 0) {
            connectionHeader
            if let state = store.state, !state.isLive {
                HStack(spacing: 8) {
                    Image(systemName: "flask")
                    Text(state.mode == "rehearsal" ? "操作演练 · 合成局面" : "模拟器回放 · 非实战建议")
                        .fontWeight(.medium)
                    Spacer(minLength: 2)
                    Button("退出") { Task { await store.send("mode", values: ["mode": "live"]) } }
                        .buttonStyle(.borderless)
                        .accessibilityIdentifier("exit-rehearsal")
                }
                .font(.system(size: 12)).padding(.horizontal, 20).padding(.vertical, 10)
                .background(.orange.opacity(0.13))
            }
            Picker("内容", selection: $tab) {
                Text("当前指引").tag("guidance")
                Text("局面核对").tag("state")
            }
            .pickerStyle(.segmented).labelsHidden().padding(.horizontal, 20).padding(.top, 18)
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if let error = store.errorMessage {
                        HStack(alignment: .top) {
                            Text(error).font(.callout).foregroundStyle(.orange)
                            Spacer(minLength: 4)
                            Button { store.errorMessage = nil } label: { Image(systemName: "xmark") }
                                .buttonStyle(.plain).help("关闭提示")
                        }
                    }
                    if tab == "guidance" {
                        GuidanceView(store: store)
                    } else {
                        StateReviewView(state: store.state)
                    }
                    if let state = store.state, !state.isLive, !state.history.isEmpty {
                        HistoryView(events: state.history)
                    }
                }
                .padding(20)
            }
        }
        .frame(minWidth: 360, minHeight: 500)
        .containerBackground(.regularMaterial, for: .window)
        .toolbar(removing: .title)
        .background(FloatingWindow(pinned: pinned, compact: compact).frame(width: 0, height: 0))
        .toolbar {
            ToolbarItemGroup(placement: .automatic) {
                Button { pinned.toggle() } label: {
                    Image(systemName: pinned ? "pin.fill" : "pin")
                }
                .help(pinned ? "取消置顶" : "窗口置顶")
                .accessibilityLabel(pinned ? "取消置顶" : "窗口置顶")
                .accessibilityIdentifier("toggle-pin")
                Button { Task { await store.send("refresh") } } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .help("重新读取局面 ⌘R").accessibilityLabel("重新读取")
                .accessibilityIdentifier("refresh-state").disabled(store.busy)
                Menu {
                    Toggle("紧凑窗口", isOn: $compact)
                    Divider()
                    Button("开始操作演练") { Task { await store.send("mode", values: ["mode": "rehearsal"]) } }
                    Button("返回真实日志") { Task { await store.send("mode", values: ["mode": "live"]) } }
                    Divider()
                    Button("导入模拟器预览…") { Task { await store.importPreview() } }
                } label: { Image(systemName: "ellipsis.circle") }
                .help("更多操作").accessibilityLabel("更多操作")
            }
        }
        .task { await store.poll() }
    }

    private var connectionHeader: some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: "rectangle.stack.fill")
                .font(.system(size: 24, weight: .medium)).foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 5) {
                Text("AutoTavern").font(.system(size: 18, weight: .semibold, design: .rounded))
                HStack(spacing: 5) {
                    Circle().fill(store.state?.clientRunning == true ? Color.green : Color.secondary)
                        .frame(width: 6, height: 6)
                    Text(store.state?.clientRunning == true ? "炉石已运行" : "等待客户端")
                    Text("·")
                    Text(store.connected ? "自动读取中" : "连接中")
                }
                .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            Spacer()
            if store.busy { ProgressView().controlSize(.mini) }
        }
        .padding(.horizontal, 20).padding(.top, 18).padding(.bottom, 16)
    }
}
