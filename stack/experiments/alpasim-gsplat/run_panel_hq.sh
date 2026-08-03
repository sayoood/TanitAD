#!/usr/bin/env bash
# STREAM C — the METRICS panel re-run on the 2026-08-03 IMPROVED render.
#
# WHY THIS EXISTS. The render-quality pass raised grad-NCC 0.2774 -> 0.3424 (+23.4 %) and
# the four closed-loop videos were re-rendered with it. The DRIVING metrics beside those
# files were not: they are the morning panel, measured on `background,road` with no sky.
# Changing the render changes what the policy SEES, so those numbers do not transfer.
# This script produces the panel that was actually measured on what the videos show.
#
# THE ONLY DIFFERENCE vs run_panel.sh is the render flags. Same scene, same checkpoints,
# same 9 starts, same 50 ticks, same code path, same scorer — so the contrast is a clean
# A/B on the render and nothing else.
#
#   RQ flags: --all-dynamic-layers --cull-scale-quantile 0.95 --sky-gain 0.3
#   `empty` deliberately does NOT get --all-dynamic-layers: it is the matched control for
#   actor presence, exactly as run_quality_videos.sh treats it, so objects-vs-empty stays
#   an A/B on actors and not on the layer set.
#
# ROLLING SHUTTER IS OFF, as in the shipped videos (161x cost). If it is ever turned on
# for the videos, this panel has to be re-run again with ROLL=1 — the same rule that made
# this script necessary.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=${1:-$HOME/cl_out_hq}
STARTS=${2:-0,17,34,51,68,85,102,119,136}
STEPS=${3:-50}
ROLL=${ROLL:-0}
mkdir -p "$OUT"

RQ="--cull-scale-quantile 0.95 --sky-gain 0.3"
[ "$ROLL" = "1" ] && RQ="$RQ --rolling-shutter"
echo "RENDER-QUALITY FLAGS (empty): $RQ"
echo "RENDER-QUALITY FLAGS (objects): $RQ --all-dynamic-layers"

declare -A CKPT=(
  [flagship-v1]=$HOME/models/flagship-v1-speedjerk/ckpt.pt
  [refc-base]=$HOME/models/refc-base/ckpt.pt
)

for ARM in flagship-v1 refc-base; do
  for COND in empty objects; do
    EXTRA="$RQ"
    [ "$COND" = "objects" ] && EXTRA="$RQ --all-dynamic-layers"
    D="$OUT/panel_${ARM}_${COND}"
    echo "=== PANEL $ARM / $COND (starts $STARTS x $STEPS) ==="
    python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
        --condition "$COND" --starts "$STARTS" --steps "$STEPS" \
        --out "$D" $EXTRA || echo "FAILED panel $ARM $COND"
  done
done
echo "PANEL_HQ_DONE"
