#!/usr/bin/env bash
# Production panel: long closed-loop videos + the metrics starts, both arms.
# Run detached on Thor, log to /tmp (NEVER a network mount).
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat:$HOME/alpasim/src/grpc
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=${1:-$HOME/cl_out}
COND=${2:-empty}
STARTS=${3:-0,17,34,51,68,85,102,119,136}
STEPS=${4:-50}
LONG_STEPS=${5:-180}
mkdir -p "$OUT"

declare -A CKPT=(
  [flagship-v1]=$HOME/models/flagship-v1-speedjerk/ckpt.pt
  [refc-base]=$HOME/models/refc-base/ckpt.pt
  [refc-xl]=$HOME/models/refc-xl/ckpt.pt
)

for ARM in flagship-v1 refc-base; do
  echo "=== $ARM / $COND : LONG rollout ($LONG_STEPS steps, with frames) ==="
  python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
      --condition "$COND" --starts 0 --steps "$LONG_STEPS" \
      --out "$OUT/long_${ARM}_${COND}" --save-video-frames || echo "FAILED long $ARM"
  echo "=== $ARM / $COND : METRICS panel (starts $STARTS x $STEPS) ==="
  python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
      --condition "$COND" --starts "$STARTS" --steps "$STEPS" \
      --out "$OUT/panel_${ARM}_${COND}" || echo "FAILED panel $ARM"
done
echo "PANEL_DONE $COND"
