#!/bin/bash
# SMOKE (equivalence check, not a result): run the chain's decisive-rollout command, FULL SIZE (2,399 windows), on the banked M2b_s0 HACC checkpoint (the very checkpoint the T1-HEADS
# causal s0 rollout used, sha256 71de3c61...), then compare the tick file with C:/Users/Admin/t1hd_run/ticks/ticks_HACC_s0_m10_causal_heads.npz.
# The command line below is the dry-run's line for a causal rollout with ckpt / tag / out-dir changed and NO --max-windows; env = the chain's.
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
TT=E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
SF=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh
OUT=E:/Projects/TanitAD-artifacts/orct1/smoke_equiv
mkdir -p "$OUT/ticks" C:/Users/Admin/orct1/smoke/equiv
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TT/stack")
export PYTHONPATH="$TT/stack;$TT/taniteval;$TT/stack/scripts"
t0=$(date +%s)
C:/Users/Admin/venvs/tanitad/Scripts/python.exe "$TT/stack/scripts/v8f_t1_eval.py" run --ckpt E:/Projects/TanitAD-artifacts/orct1/smoke_art/M2b_s0_orc/ckpt_last.pt --tag EQUIV_s0_m10_causal_heads \
  --tracks "$SF/inputs/native_tracks.pt" --eval-cache "$SF/cache" --eval-grid "$TT/$PK/raw/eval_grid_windows.json" --eval-join "$TT/$PK/raw/join_eval141.npz" --out-dir "$OUT" \
  --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$OUT/ticks" --dump-heads --device cpu \
  > C:/Users/Admin/orct1/smoke/equiv/equiv.out 2>&1 < /dev/null
echo "rc=$? wall_s=$(( $(date +%s) - t0 ))" >> C:/Users/Admin/orct1/smoke/equiv/equiv.out
