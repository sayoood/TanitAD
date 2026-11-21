#!/usr/bin/env bash
# refcv3 closed-loop panel + the two historical arms re-run through the PATCHED
# harness on the NEW code tree. Protocol IDENTICAL to run_panel_hq.sh: same scene,
# same 9 starts, same 50 ticks, same render flags, rolling shutter OFF.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl_v3/stack:$HOME/tanitad_cl_v3/stack/scripts:$HOME/tanitad_cl_v3/taniteval:$HOME/tanitad_cl_v3/taniteval/tools:$HOME/tanitad_cl_v3/stack/experiments/nurec-gsplat:$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat"
SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=$HOME/cl_out_v3
STARTS=0,17,34,51,68,85,102,119,136
STEPS=50
RQ="--cull-scale-quantile 0.95 --sky-gain 0.3"
mkdir -p "$OUT"
run () {  # arm ckpt extra
  local ARM=$1 CK=$2; shift 2
  local D="$OUT/panel_${ARM}_empty"
  echo "=== PANEL $ARM (starts $STARTS x $STEPS) ==="
  python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "$CK" \
      --condition empty --starts "$STARTS" --steps "$STEPS" --out "$D" $RQ "$@" \
      && echo "ZZOK_${ARM}ZZ" || echo "ZZFAIL_${ARM}ZZ"
}
run refcv3      $HOME/models/refcv3-b1-v72-40284/ckpt.pt
run refc-base   $HOME/models/refc-base/ckpt.pt
run flagship-v1 $HOME/models/flagship-v1-speedjerk/ckpt.pt
echo "ZZPANEL_V3_DONEZZ"
