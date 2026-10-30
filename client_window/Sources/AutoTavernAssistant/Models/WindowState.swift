import Foundation

struct Connection: Decodable, Sendable {
    let url: URL
    let token: String
}

struct WindowState: Decodable, Sendable {
    let revision: String?
    let gameStatus: String?
    let recentOperations: [ObservedOperation]?
    let mode: String
    let modeLabel: String
    let clientRunning: Bool?
    let clientBuild: String?
    let clientScene: String?
    let stateTitle: String
    let stateDetail: String
    let readStatus: String
    let freshness: String
    let modelReady: Bool
    let modelStatus: ModelStatus?
    let gameKey: String?
    let lobbyRaces: [String]?
    let needsLobbyRaces: Bool?
    let liveReady: Bool
    let proposal: Proposal?
    let message: String?
    let canNextFrame: Bool
    let nextFrameLabel: String
    let cards: [VisibleCard]
    let facts: [StateFact]
    let blockers: [String]
    let history: [FeedbackEvent]
    let polledAt: Double

    var isLive: Bool { mode == "live" }
}

struct ModelStatus: Decodable, Sendable {
    let title: String
    let detail: String
}

struct ObservedOperation: Decodable, Sendable, Identifiable {
    let id: String
    let title: String
    let detail: String
}

struct Proposal: Decodable, Sendable, Identifiable {
    let id: String
    let frameId: String
    let scope: String
    let title: String
    let instruction: String
    let references: [CardReference]
    let cost: ActionCost?
    let expiresAt: Double
    let liveReady: Bool
}

struct CardReference: Decodable, Sendable {
    let zone: String
    let position: Int
    let name: String
    let attack: Int?
    let health: Int?
}

struct ActionCost: Decodable, Sendable {
    let amount: Int
    let currency: String
    var label: String { "\(amount) \(["gold": "金币", "health": "生命值", "chronum": "时空币"][currency] ?? currency)" }
}

struct VisibleCard: Decodable, Sendable, Identifiable {
    let id: String
    let name: String
    let detail: String
    let cardId: String?
    let group: String?
    let groupTitle: String?
}

struct StateFact: Decodable, Sendable {
    let label: String
    let value: String
}

struct FeedbackEvent: Decodable, Sendable, Identifiable {
    let id: String
    let at: String
    let mode: String
    let title: String
    let outcome: String
    let executionVerified: Bool
}
