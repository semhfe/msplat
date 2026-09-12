#!/bin/bash
# Regression check for the seed-fog collapse (tnt_meetingroom, 2026-09-12).
# An indoor capture whose SfM cloud has isolated seeds near the camera path collapsed
# to ~11 dB: full-spacing seed scales at opacity 0.5 stacked into an opaque fog that
# saturated every pixel (T <= 1e-4) after ~85 splats, so 76 % of the population got
# exactly zero gradient, the opacity regularizer then killed them and relocation cloned
# the fog. The reference init (3dgs-mcmc create_from_pcd: 0.1 x kNN spacing) keeps the
# scene translucent. This script fails if the collapse comes back.
#
# usage: tests/smoke_seed_fog.sh [msplat-binary] [dataset-dir]
set -u
MS=${1:-"$(dirname "$0")/../build/msplat"}
DS=${2:-"$HOME/Downloads/newcorpus/ws/tnt_meetingroom"}
LOG=$(mktemp -t seedfog)
"$MS" "$DS" -d 2 --cap-max 250000 -n 3000 --eval --test-every 8 --no-seed-decimation --cull-radius 0 -o /dev/null > "$LOG" 2>&1
psnr=$(grep -oE '^  PSNR: +-?[0-9.]+' "$LOG" | grep -oE '[-0-9.]+$')
read -r active junk < <(grep -oE 'MCMC step=2000 active=[0-9]+ .*junkdc=[0-9]+' "$LOG" | sed -E 's/.*active=([0-9]+).*junkdc=([0-9]+)/\1 \2/')
junkfrac=$(awk -v a="${active:-1}" -v j="${junk:-0}" 'BEGIN{printf "%.3f", j/a}')
echo "seed-fog smoke: PSNR@3000=${psnr:-FAIL} junk@2000=${junkfrac} (log: $LOG)"
# Collapse signature: PSNR 10-12 dB, junk >= 0.64 at step 2000. Healthy: >= 20 dB, junk <= 0.3.
awk -v p="${psnr:-0}" -v j="$junkfrac" 'BEGIN{exit !(p >= 20.0 && j <= 0.30)}' && echo PASS || { echo FAIL; exit 1; }
