import AppKit
import SwiftUI

// SwiftUI owns the scene; this bridge only applies floating/Space behavior.
struct FloatingWindow: NSViewRepresentable {
    let pinned: Bool
    let compact: Bool

    func makeNSView(context: Context) -> WindowProbe { WindowProbe() }
    func updateNSView(_ view: WindowProbe, context: Context) {
        view.pinned = pinned
        view.compact = compact
        view.apply()
    }
}

final class WindowProbe: NSView {
    var pinned = true
    var compact = false
    private var appliedCompact: Bool?
    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        apply()
    }
    func apply() {
        guard let window else { return }
        window.level = pinned ? .floating : .normal
        window.collectionBehavior = pinned ? [.canJoinAllSpaces, .fullScreenAuxiliary] : [.managed]
        if appliedCompact != compact {
            let shouldResize = appliedCompact != nil || compact
            appliedCompact = compact
            if shouldResize {
                let oldFrame = window.frame
                let content = NSRect(origin: .zero, size: compact ? NSSize(width: 360, height: 520) : NSSize(width: 420, height: 700))
                var frame = window.frameRect(forContentRect: content)
                frame.origin = NSPoint(x: oldFrame.maxX - frame.width, y: oldFrame.maxY - frame.height)
                if let visible = window.screen?.visibleFrame {
                    frame.size.height = min(frame.height, visible.height)
                    frame.origin.x = max(visible.minX, min(frame.origin.x, visible.maxX - frame.width))
                    frame.origin.y = max(visible.minY, min(frame.origin.y, visible.maxY - frame.height))
                }
                window.setFrame(frame, display: true, animate: false)
            }
        }
    }
}
