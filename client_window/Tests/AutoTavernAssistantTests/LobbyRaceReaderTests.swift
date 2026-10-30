import AppKit
import XCTest
import Vision
@testable import AutoTavernAssistant

final class LobbyRaceReaderTests: XCTestCase {
    func testBannerMustContainHeroHeadingAndExactlyFiveUniqueRaces() {
        XCTAssertNil(LobbyRaceReader.extract(["亡灵 元素 龙 恶魔 鱼人"]))
        XCTAssertNil(LobbyRaceReader.extract(["选择一个英雄", "亡灵 元素 龙 恶魔"]))
        XCTAssertEqual(LobbyRaceReader.extract(["选择一个英雄", "畸变 元素 龙 恶魔 鱼人"]),
                       ["ABERRATION", "DEMON", "DRAGON", "ELEMENTAL", "MURLOC"])
        XCTAssertNil(LobbyRaceReader.extract(["选择一个英雄", "亡灵 元素 龙 恶魔 鱼人 机械"]))
    }

    func testVisionReadsHeroBannerPixels() throws {
        let image = NSImage(size: NSSize(width: 1600, height: 900))
        image.lockFocus()
        NSColor.black.setFill()
        NSRect(x: 0, y: 0, width: 1600, height: 900).fill()
        let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 42), .foregroundColor: NSColor.white]
        ("选择一个英雄" as NSString).draw(at: NSPoint(x: 400, y: 750), withAttributes: attributes)
        ("亡灵 元素 野猪人 鱼人 龙" as NSString).draw(at: NSPoint(x: 330, y: 650), withAttributes: attributes)
        image.unlockFocus()
        let cg = try XCTUnwrap(image.cgImage(forProposedRect: nil, context: nil, hints: nil))
        let probe = VNRecognizeTextRequest()
        probe.recognitionLanguages = ["zh-Hans", "en-US"]
        try VNImageRequestHandler(cgImage: cg).perform([probe])
        let candidates = (probe.results ?? []).compactMap { $0.topCandidates(1).first }.map { "\($0.string):\($0.confidence)" }
        let lines = try LobbyRaceReader.recognizeLines(cg)
        XCTAssertEqual(LobbyRaceReader.extract(lines), ["DRAGON", "ELEMENTAL", "MURLOC", "QUILBOAR", "UNDEAD"], "OCR lines: \(lines); unfiltered: \(candidates)")
    }
}
