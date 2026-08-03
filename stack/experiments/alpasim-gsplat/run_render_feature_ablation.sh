#!/usr/bin/env bash
# STREAM C — WHICH render feature moves the policy?
#
# MEASURED first, then this script: on the improved render flagship v1's closed-loop ADE
# rose +6.05 m [4.13, 7.74] while REF-C's moved -0.08 m [-0.29, +0.08], and the
# morning-config re-run reproduced the morning rollouts EXACTLY (0.0 m on 450/450
# windows, both arms, all 19 metrics). So the whole 6 m is the render — but "the render"
# was two changes at once on `empty` (scale cull + gated sky).
#
# This is the 2x2 that separates them. `background,road` and `HQ` already exist, so only
# the two single-factor cells are missing:
#
#            no cull        cull 0.95
#   no sky   = MORNING      = CULL
#   sky 0.3  = SKY          = HQ
#
# Also runs the `objects` MORNING config, which the empty-condition control does not
# cover: `closedloop_drive.py` gained `act["tracks"].t0_us = float(t_us)` today, and that
# line only executes with actors attached. Until it is measured, an objects-condition
# morning-vs-HQ difference is NOT attributable to the render.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=${1:-$HOME/cl_out_ablate}
STARTS=${2:-0,17,34,51,68,85,102,119,136}
STEPS=${3:-50}
mkdir -p "$OUT"

declare -A CKPT=(
  [flagship-v1]=$HOME/models/flagship-v1-speedjerk/ckpt.pt
  [refc-base]=$HOME/models/refc-base/ckpt.pt
)

run () {  # arm cell flags...
  local ARM=$1 CELL=$2; shift 2
  echo "=== ABLATE $ARM / $CELL : $* ==="
  python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
      --condition empty --starts "$STARTS" --steps "$STEPS" \
      --out "$OUT/${CELL}_${ARM}_empty" "$@" || echo "FAILED $ARM $CELL"
}

for ARM in flagship-v1 refc-base; do
  run "$ARM" cull --cull-scale-quantile 0.95
  run "$ARM" sky  --sky-gain 0.3
done

# the `objects` morning config — the control the empty run cannot provide
for ARM in flagship-v1 refc-base; do
  echo "=== OBJECTS MORNING-CONFIG CONTROL $ARM ==="
  python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
      --condition objects --starts "$STARTS" --steps "$STEPS" \
      --out "$OUT/objmorn_${ARM}_objects" || echo "FAILED $ARM objmorn"
done
echo "ABLATION_DONE"
