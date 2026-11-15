#!/usr/bin/env bash
# smoke_cpu.sh <name> <dist-rule> [pinned]  -- CPU smoke on 16 windows of the real evaluation inputs of the dev box (checkpoint P_s0, no lead head).
# name: output folder under tmp/smoke; dist-rule: off|label|never|push|plumb; third argument "pinned" runs the pinned tool directly (no wrapper).
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
TREE=C:/Users/Admin/v8proof_c/leadsup/tree_patch
HERE=F:/TanitAD-scratch/v8proof/distrule
RAW="E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
OUT=$HERE/tmp/smoke/$1
mkdir -p "$OUT"
ARGS=(run --ckpt C:/Users/Admin/v8proof_c/ckpt/P_s0/ckpt_last.pt --tag S --tracks E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/native_tracks.pt
  --eval-cache E:/Projects/TanitAD-artifacts/sfresh_inputs/cache --eval-grid "$RAW/eval_grid_windows.json" --eval-join "$RAW/join_eval141.npz"
  --out-dir "$OUT" --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --batch 16 --device cpu --no-analyze
  --max-windows 16 --dump-ticks "$OUT/ticks")
if [ "${3:-}" = pinned ]; then
  $PY "$TREE/stack/scripts/v8f_t1_eval.py" "${ARGS[@]}"
else
  EXTRA=()
  [ "$2" = off ] || EXTRA=(--rule-log "$OUT/ticks/rule_log_S.npz" --floor "${FLOOR:-card}")
  $PY "$HERE/v8f_t1_eval_rule.py" --tree "$TREE" --dist-rule "$2" "${EXTRA[@]}" "${ARGS[@]}"
fi
