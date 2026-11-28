#!/bin/bash
# loader_test.sh -- does the T1 / T0 eval loader build a DAgger checkpoint, and does the tree matter?  (CPU, 24 windows, outputs under dagt1/loader_test; nothing else is written)
#   1. a COPY of a DAgger-trainer smoke checkpoint (dt/bt/r1/test_p_one_swaps_exactly_the_e0/p1/ckpt_last.pt: written by the DAgger trainer, p = 1, 'dagger' + 'dagger_totals' keys) is rolled by
#      the REAL v8f_t1_eval.py (the chain's exact flags, --max-windows 24) from each tree, and scored by the REAL v8f_mini_eval.py (--ages fresh);
#   2. a real ORC-0 checkpoint (a COPY) is rolled from both trees and the tick files are compared array by array.
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
R=E:/Projects/TanitAD-artifacts; H=$R/dagt1/loader_test; Hu=/e/Projects/TanitAD-artifacts/dagt1/loader_test
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
SF=E:/Projects/TanitAD-artifacts/sfresh_inputs
TRACKS=$SF/inputs/native_tracks.pt; CACHE=$SF/cache; V9=$SF/inputs/v9; P10=$SF/inputs/p10/p10_prior.pt
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TMP=$R/dagt1/tmp TEMP=$R/dagt1/tmp
mkdir -p "$Hu/out" "$Hu/ck" "$R/dagt1/tmp"
DAGCK_SRC="$R/dt/bt/r1/test_p_one_swaps_exactly_the_e0/p1/ckpt_last.pt"; ORCCK_SRC="$R/v8f-orecover/M2b_s0_orc0/ckpt_last.pt"
cp -f "$DAGCK_SRC" "$Hu/ck/dag_smoke_copy.pt"; [ -s "$Hu/ck/orc0_s0_copy.pt" ] || cp "$ORCCK_SRC" "$Hu/ck/orc0_s0_copy.pt"
sha256sum "$DAGCK_SRC" "$Hu/ck/dag_smoke_copy.pt" | cut -c1-70
run_t1() {   # run_t1 <tree> <label> <ckpt> <tag>
  local TT=$1 lab=$2 ck=$3 tag=$4 TTu; TTu=$(cygpath -u "$TT")
  export TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TTu/stack") PYTHONPATH="$TT/stack;$TT/taniteval;$TT/stack/scripts"
  mkdir -p "$Hu/out/$lab/ticks"
  "$PY" "$TT/stack/scripts/v8f_t1_eval.py" run --ckpt "$ck" --tag "$tag" --tracks "$TRACKS" --eval-cache "$CACHE" --eval-grid "$TT/$PK/raw/eval_grid_windows.json" --eval-join "$TT/$PK/raw/join_eval141.npz" \
    --out-dir "$H/out/$lab" --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$H/out/$lab/ticks" --dump-heads --device cpu --max-windows 24 > "$Hu/out/$lab/$tag.out" 2>&1
  echo "T1 $lab $tag rc=$?  $(tail -1 "$Hu/out/$lab/$tag.out" | cut -c1-160)"
}
run_t0() {   # run_t0 <tree> <label> <ckpt> <tag>
  local TT=$1 lab=$2 ck=$3 tag=$4 TTu; TTu=$(cygpath -u "$TT")
  export TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TTu/stack") PYTHONPATH="$TT/stack;$TT/taniteval;$TT/stack/scripts"
  mkdir -p "$Hu/out/$lab/t0"
  "$PY" "$TT/stack/scripts/v8f_mini_eval.py" run --ckpt "$ck" --tag "$tag" --ages fresh --v9-eval "$V9/v9_labels_eval139.npz" --lead-eval "$V9/lead_eval139.npz" --speed-eval "$V9/refcv6_speed_max_v8_eval.jsonl" \
    --kin-mode yaw --p10 "$P10" --out-dir "$H/out/$lab/t0" --eval-cache "$CACHE" --tracks "$TRACKS" --eval-grid "$TT/$PK/raw/eval_grid_windows.json" --eval-kin "$TT/$PK/raw/eval141_kin.npz" \
    --eval-join "$TT/$PK/raw/join_eval141.npz" --project --families > "$Hu/out/$lab/$tag.t0.out" 2>&1
  echo "T0 $lab $tag rc=$?  $(tail -1 "$Hu/out/$lab/$tag.t0.out" | cut -c1-160)"
}
DI=$R/di/tree; T81=$R/v8f-t1heads-run/tool_81aa3e36
case "${1:-all}" in
  dag_t1)  run_t1 "$DI" di "$H/ck/dag_smoke_copy.pt" LT_DAGSMOKE; run_t1 "$T81" tool81 "$H/ck/dag_smoke_copy.pt" LT_DAGSMOKE ;;
  dag_t0)  run_t0 "$DI" di "$H/ck/dag_smoke_copy.pt" LT_DAGSMOKE-fresh ;;
  orc_t1)  run_t1 "$DI" di "$H/ck/orc0_s0_copy.pt" LT_ORC0; run_t1 "$T81" tool81 "$H/ck/orc0_s0_copy.pt" LT_ORC0 ;;
  all)     "$0" dag_t1; "$0" dag_t0; "$0" orc_t1 ;;
esac
