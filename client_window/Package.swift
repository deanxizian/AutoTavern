// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AutoTavernAssistant",
    platforms: [.macOS(.v15)],
    products: [.executable(name: "AutoTavernAssistant", targets: ["AutoTavernAssistant"])],
    targets: [.executableTarget(name: "AutoTavernAssistant"),
        .testTarget(name: "AutoTavernAssistantTests", dependencies: ["AutoTavernAssistant"])]
)
