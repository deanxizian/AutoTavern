import AppKit
import Observation
import UniformTypeIdentifiers

@MainActor @Observable final class SessionStore {
    var state: WindowState?
    var connected = false
    var busy = false
    var errorMessage: String?
    private var api: WindowAPI?
    private var generation = 0

    init() {
        let args = CommandLine.arguments
        guard let index = args.firstIndex(of: "--connection"), args.indices.contains(index + 1),
              let data = try? Data(contentsOf: URL(fileURLWithPath: args[index + 1])),
              let connection = try? JSONDecoder().decode(Connection.self, from: data) else {
            errorMessage = "请使用项目的“打开对战助手.command”启动窗口。"
            return
        }
        api = WindowAPI(connection: connection)
    }

    func poll() async {
        while !Task.isCancelled {
            if !busy { await load() }
            do { try await Task.sleep(for: .milliseconds(50)) } catch { break }
        }
    }

    func load() async {
        guard let api else { return }
        let requestedGeneration = generation
        do {
            let latest = try await api.request("state")
            guard requestedGeneration == generation, !busy else { return }
            if state == nil || latest.revision == nil || state?.revision != latest.revision {
                state = latest
            }
            if errorMessage == "本机连接中断，指引已暂停。请重新读取或重新打开助手。" {
                errorMessage = nil
            }
            connected = true
        } catch {
            guard requestedGeneration == generation, !busy else { return }
            connected = false
            errorMessage = "本机连接中断，指引已暂停。请重新读取或重新打开助手。"
        }
    }

    func send(_ route: String, values: [String: String] = [:]) async {
        guard let api, !busy else { return }
        generation += 1
        busy = true
        defer { busy = false }
        do {
            let data = try JSONSerialization.data(withJSONObject: values)
            state = try await api.request(route, body: data)
            connected = true
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
            // Any failed mutation hides old instructions until a fresh read succeeds.
            connected = false
        }
    }

    func importPreview() async {
        guard let api, !busy else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        panel.message = "导入含 view 和 action 的模拟器预览文件。实战建议自动从当前对局读取。"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        generation += 1
        busy = true
        defer { busy = false }
        do {
            let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
            guard let size = attributes[.size] as? NSNumber, size.intValue <= 4 * 1024 * 1024 else {
                throw APIError(message: "预览文件超过 4 MB，请仅导出当前决策局面。")
            }
            state = try await api.request("preview", body: Data(contentsOf: url))
            connected = true
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}
