import AppKit
import ScreenCaptureKit
import Vision

/// Reads only the visible hero-selection banner in the Hearthstone window.
/// Images stay in memory; no game input, OCR card parsing or remote service.
enum LobbyRaceReader {
    static let names = [("亡灵", "UNDEAD"), ("元素", "ELEMENTAL"), ("野猪人", "QUILBOAR"),
        ("鱼人", "MURLOC"), ("龙", "DRAGON"), ("野兽", "BEAST"), ("恶魔", "DEMON"),
        ("机械", "MECHANICAL"), ("海盗", "PIRATE"), ("异变", "ABERRATION"),
        ("畸变", "ABERRATION"), ("纳迦", "NAGA")]

    static func extract(_ lines: [String]) -> [String]? {
        guard lines.contains(where: { $0.replacingOccurrences(of: " ", with: "").contains("选择一个英雄") }) else { return nil }
        for line in lines {
            let matches = Set(names.filter { line.contains($0.0) }.map(\.1))
            if matches.count == 5 { return matches.sorted() }
        }
        return nil
    }

    static func read() async throws -> [String]? {
        let content = try await SCShareableContent.excludingDesktopWindows(true, onScreenWindowsOnly: true)
        guard let window = content.windows.first(where: {
            $0.owningApplication?.applicationName == "Hearthstone" && $0.frame.width > 500 && $0.frame.height > 400
        }) else { return nil }
        let filter = SCContentFilter(desktopIndependentWindow: window)
        let config = SCStreamConfiguration()
        let scale = min(2.0, 1920.0 / window.frame.width)
        config.width = Int(window.frame.width * scale)
        config.height = Int(window.frame.height * scale)
        config.showsCursor = false
        let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        return try recognize(image)
    }

    static func recognize(_ image: CGImage) throws -> [String]? {
        extract(try recognizeLines(image))
    }

    static func recognizeLines(_ image: CGImage) throws -> [String] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["zh-Hans", "en-US"]
        request.usesLanguageCorrection = false
        // Only the top banner can establish the lobby list. A minion's printed
        // tribe elsewhere in the screen is not evidence of the full lobby pool.
        request.regionOfInterest = CGRect(x: 0.15, y: 0.60, width: 0.75, height: 0.36)
        try VNImageRequestHandler(cgImage: image).perform([request])
        let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first }
            // Vision assigns 0.5 to correctly read Chinese race lists. Require
            // the hero heading and exactly five distinct names as well.
            .filter { $0.confidence >= 0.5 }.map(\.string)
        return lines
    }
}
