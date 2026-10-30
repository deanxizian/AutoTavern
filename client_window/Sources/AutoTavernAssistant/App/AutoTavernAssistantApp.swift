import AppKit
import SwiftUI

@main struct AutoTavernAssistantApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @State private var store = SessionStore()
    @AppStorage("assistant.compact") private var compact = false

    var body: some Scene {
        Window("AutoTavern 对战助手", id: "assistant") {
            ContentView(store: store)
        }
        .defaultSize(width: 420, height: 700)
        .defaultWindowPlacement { _, context in
            let rect = context.defaultDisplay.visibleRect
            let size = CGSize(width: min(420, rect.width - 32), height: min(700, rect.height - 48))
            return WindowPlacement(CGPoint(x: rect.maxX - size.width - 20, y: rect.minY + 32), size: size)
        }
        .windowResizability(.contentMinSize)
        .commands {
            CommandMenu("对战助手") {
                Toggle("紧凑窗口", isOn: $compact)
                    .keyboardShortcut("k", modifiers: [.command, .shift])
                Divider()
                Button("重新读取") { Task { await store.send("refresh") } }
                    .keyboardShortcut("r", modifiers: .command)
                Button("开始操作演练") { Task { await store.send("mode", values: ["mode": "rehearsal"]) } }
                Button("返回真实日志") { Task { await store.send("mode", values: ["mode": "live"]) } }
                Divider()
                Button("导入模拟器预览…") { Task { await store.importPreview() } }
            }
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
