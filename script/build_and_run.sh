#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="AutoTavernAssistant"
BUILD_DIR="$ROOT_DIR/runs/client-window/build"
APP_BUNDLE="$ROOT_DIR/runs/client-window/$APP_NAME.app"
CLIENT_PYTHON="$ROOT_DIR/runs/client-env/bin/python"

case "$MODE" in
  run|--verify|--build-only) ;;
  *) echo "usage: $0 [run|--verify|--build-only]" >&2; exit 2 ;;
esac
if [ ! -x "$CLIENT_PYTHON" ]; then
  echo "Missing runs/client-env; follow docs/CLIENT_STATE.md to install the log parser." >&2
  exit 1
fi
if [ "$MODE" != "--build-only" ]; then
  pkill -x "$APP_NAME" >/dev/null 2>&1 || true
fi
mkdir -p "$BUILD_DIR" "$APP_BUNDLE/Contents/MacOS"
bash "$ROOT_DIR/script/build_client_native.sh"
export CLANG_MODULE_CACHE_PATH="$BUILD_DIR/ModuleCache"
swift build --package-path "$ROOT_DIR/client_window" --scratch-path "$BUILD_DIR"
BIN_DIR="$(swift build --package-path "$ROOT_DIR/client_window" --scratch-path "$BUILD_DIR" --show-bin-path)"
cp "$BIN_DIR/$APP_NAME" "$APP_BUNDLE/Contents/MacOS/$APP_NAME"
# Link only Apple frameworks; load the installed HearthSim-signed reader at
# runtime. Never copy a vendor binary into the application or repository.
xcrun clang++ -std=c++17 -fobjc-arc -O2 -framework AppKit -framework Security \
  "$ROOT_DIR/tools/client_mirror/main.mm" -o "$APP_BUNDLE/Contents/MacOS/AutoTavernMirrorReader"
/usr/bin/codesign --force --sign - --entitlements "$ROOT_DIR/tools/client_mirror/reader.entitlements" \
  "$APP_BUNDLE/Contents/MacOS/AutoTavernMirrorReader"
# Remove the obsolete generated helper when upgrading an existing app bundle.
rm -f "$APP_BUNDLE/Contents/MacOS/AutoTavernRaceReader"
"$CLIENT_PYTHON" - "$APP_BUNDLE/Contents/Info.plist" <<'PY'
import plistlib, sys
from pathlib import Path
Path(sys.argv[1]).write_bytes(plistlib.dumps({
    'CFBundleExecutable': 'AutoTavernAssistant',
    'CFBundleIdentifier': 'com.autotavern.local-assistant',
    'CFBundleName': 'AutoTavern 对战助手',
    'CFBundleDisplayName': 'AutoTavern 对战助手',
    'CFBundlePackageType': 'APPL',
    'CFBundleShortVersionString': '0.1.0',
    'CFBundleVersion': '1',
    'LSMinimumSystemVersion': '15.0',
    'NSPrincipalClass': 'NSApplication',
    'NSAppTransportSecurity': {'NSAllowsLocalNetworking': True},
}))
PY
# Seal the staged bundle after writing Info.plist. SwiftPM's executable-only
# ad-hoc signature does not cover the app resources used by Launch Services.
/usr/bin/codesign --force --sign - "$APP_BUNDLE"
/usr/bin/codesign --verify --strict "$APP_BUNDLE"
if [ "$MODE" == "--build-only" ]; then
  echo "Built $APP_BUNDLE"
  exit 0
fi
cd "$ROOT_DIR"
if [ "$MODE" == "--verify" ]; then
  exec "$CLIENT_PYTHON" tools/client_window.py --bundle "$APP_BUNDLE" --verify
fi
exec "$CLIENT_PYTHON" tools/client_window.py --bundle "$APP_BUNDLE"
