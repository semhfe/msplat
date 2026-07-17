#!/bin/bash
# Build libmsplat_core.a + default.metallib for iOS devices (arm64).
# Usage: scripts/build-ios.sh [output-dir]
# Artifacts land in build-ios/ and are copied to output-dir when given.
set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="${1:-}"

cmake -B build-ios \
    -DCMAKE_SYSTEM_NAME=iOS \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=16.0 \
    -DMSPLAT_METAL_SDK=iphoneos \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-ios -j --target msplat_core metallib

echo "=== Built ==="
lipo -info build-ios/libmsplat_core.a
ls -la build-ios/default.metallib

if [ -n "$OUT_DIR" ]; then
    mkdir -p "$OUT_DIR"
    cp build-ios/libmsplat_core.a "$OUT_DIR/"
    cp build-ios/default.metallib "$OUT_DIR/"
    echo "=== Copied to $OUT_DIR ==="
fi
