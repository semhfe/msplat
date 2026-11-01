#!/bin/zsh
# Runs tests/q_gpu_harness.mm against a pinned fixture (P2-M1). The expected manifest sha256 is fixed here; the harness
# exits 1 if the fixture's manifest.sha256, its 36-file inventory or any file hash differs.
# Usage: tests/q_harness_run.sh <reference|mutant-check> <new|old>
#   new: v7 fixture (time-varying mask)   old: pre-v7 fixture (static mask), kept for the mutation check
# Take the GPU lock (~/.msplat-gpu-lock) before calling this; the script does not.
set -u
cd "${0:A:h}/.."
NEW_DIR=/Volumes/fe-ssd-2/hiz_q_fixture_2026-11-01-v7
NEW_SHA=9028160a9f863f9aae80cf8653e9ef9b6e109861998a89acbefc9494bd1ab162
OLD_DIR=/Volumes/fe-ssd-2/hiz_q_fixture_2026-11-01
OLD_SHA=4e7d056f08d344404e63252b65c3443ac825dc5d20b5f1ceb4c8e424a7d90ae7
if [[ $# -ne 2 ]]; then echo "usage: $0 <reference|mutant-check> <new|old>" >&2; exit 2; fi
case "$2" in
  new) DIR=$NEW_DIR; SHA=$NEW_SHA ;;
  old) DIR=$OLD_DIR; SHA=$OLD_SHA ;;
  *) echo "usage: $0 <reference|mutant-check> <new|old>" >&2; exit 2 ;;
esac
exec build-q/q_gpu_harness --mode "$1" --fixture "$DIR" --expect-manifest "$SHA"
