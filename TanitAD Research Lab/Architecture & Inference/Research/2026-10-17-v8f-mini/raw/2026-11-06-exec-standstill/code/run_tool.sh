#!/usr/bin/env bash
# run_tool.sh TOOL TAG CKPT OUTROOT GRID [extra tool flags...]
# One T1-kin rollout with the banked C0 flags (--smooth sg72 --scene logged, replan 10, causal bucket, ticks + heads), on GRID, CPU only.
# Writes OUTROOT/TAG/{dump_TAG,ticks,t1v8f_TAG.json,exec_ch_TAG.npz}; stdout in OUTROOT/TAG.out. Exit code = the tool's.
set -u
TOOL=$1; TAG=$2; CKPT=$3; OUTROOT=$4; GRID=$5; shift 5
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
TH=${EXEC_TH:-12}
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUTROOT/$TAG"
"$PY" -I "$TOOL" run --ckpt "$CKPT" --tag "$TAG" \
  --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
  --eval-grid "$GRID" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUTROOT/$TAG" \
  --replan-steps 10 --nav deadreckon --schedule periodic --limit causal \
  --dump-ticks "$OUTROOT/$TAG/ticks" --dump-heads --device cpu --smooth sg72 --scene logged "$@" < /dev/null > "$OUTROOT/$TAG.out" 2>&1
rc=$?
echo "rc=$rc" >> "$OUTROOT/$TAG.out"
exit $rc
