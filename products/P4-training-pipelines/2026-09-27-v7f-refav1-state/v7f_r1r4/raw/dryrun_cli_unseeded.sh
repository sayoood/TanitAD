#!/usr/bin/env bash
# CPU dry-run smoke of train_v6_staged.py (synthetic tensors, tiny geometry, stage S-T, 4 steps):
# the flags OFF vs ON, the per-step loss terms, and the R1b cap smoke. NOTHING here is quotable.
set -u
STACK=${V7F_STACK:-C:/Users/Admin/v7f_r1r4/stack}
OUT=${V7F_SMOKE_OUT:-C:/Users/Admin/v7f_r1r4/smoke}
export PYTHONPATH="$STACK;${V7F_TANITEVAL:-C:/Users/Admin/v7f_r1r4/taniteval}" PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
PY=${V7F_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
GEOM="--in-channels 3 --frame-h 32 --frame-w 32 --enc-dim 32 --enc-depth 1 --enc-heads 2 --readout-grid 4 --readout-dim 8 --pred-dim 32 --pred-depth 1 --pred-heads 2 --window 4 --horizons 1 --d-tac 32 --d-str 16 --d-goal-embed 16 --adapter-hidden 32 --sigreg-slices 8 --dry-steps 4 --dry-batch 4 --dry-k 12 --seed 0"
for arm in off r4_e2e r4_detached r1a r1b all_e2e; do
  case $arm in
    off) F="";;
    r4_e2e) F="--tac-op-cond e2e";;
    r4_detached) F="--tac-op-cond detached";;
    r1a) F="--max-speed-input-v6";;
    r1b) F="--plan-vmax-cap";;
    all_e2e) F="--tac-op-cond e2e --max-speed-input-v6 --plan-vmax-cap";;
  esac
  rm -rf "$OUT/$arm"
  $PY $STACK/scripts/train_v6_staged.py --stage S-T --out "$OUT/$arm" --dry-run $GEOM $F > "$OUT/$arm.log" 2>&1
  echo "$arm rc=$?"
done
