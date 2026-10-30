#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN="$ROOT_DIR/runs/model-build/rustup/toolchains/stable-aarch64-apple-darwin/bin"
if [ -x "$TOOLCHAIN/cargo" ]; then
  export PATH="$TOOLCHAIN:$PATH"
  export CARGO_HOME="$ROOT_DIR/runs/model-build/cargo"
fi
cargo build --release --manifest-path "$ROOT_DIR/tools/client_native/Cargo.toml" --target-dir "$ROOT_DIR/runs/client-native"
