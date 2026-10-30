import Foundation

struct WindowAPI: Sendable {
    let connection: Connection

    func request(_ route: String, body: Data? = nil) async throws -> WindowState {
        guard connection.url.host == "127.0.0.1", connection.url.scheme == "http" else {
            throw APIError(message: "助手只连接本机日志服务。")
        }
        var request = URLRequest(url: connection.url.appendingPathComponent("api/" + route))
        request.timeoutInterval = 3
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("Bearer " + connection.token, forHTTPHeaderField: "Authorization")
        if let body {
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = body
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            let failure = try? JSONDecoder().decode(APIError.self, from: data)
            throw failure ?? APIError(message: "局面服务未响应，请重新读取。")
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(WindowState.self, from: data)
    }
}

struct APIError: Error, Decodable, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}
