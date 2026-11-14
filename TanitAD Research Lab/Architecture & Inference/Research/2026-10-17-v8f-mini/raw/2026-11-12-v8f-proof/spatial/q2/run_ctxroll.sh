#!/usr/bin/env bash
# CTXROLL (PREREG_V8F_PROOF_ADD17 + note N1): ctx_roll.py, then the r-b lead read (rb_trained.py, unchanged) on each of the three rollouts, the pairs and the change form.
#   bash run_ctxroll.sh ARM [ARM ...]      env: CK_ROOT (checkpoints), LAT_DIR (pixel read run folder), ST_DIR (states npz), OUT (big outputs), TREE
# Serial, CPU. Deletes nothing. Small result files are copied to spatial/q2/ctx/.
W=/f/TanitAD-scratch/v8proof
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
CK_ROOT=${CK_ROOT:-F:/TanitAD-scratch/v8proof/spatial/ckpt}
LAT_DIR=${LAT_DIR:-C:/Users/Admin/v8proof_c/latvis_gw}
ST_DIR=${ST_DIR:-F:/TanitAD-scratch/v8proof/spatial/s1states}
OUT=${OUT:-C:/Users/Admin/v8proof_c/ctxroll}
TREE=${TREE:-$LAT_DIR/tree}
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
mkdir -p "$OUT/rb" "$W/spatial/q2/ctx"
cd "$W/spatial/q2" || exit 1
for ARM in "$@"; do
  if [ ! -f "$OUT/LATENTS_CTX_${ARM}_R3h.npz" ]; then
    "$PY" ctx_roll.py --arm "$ARM" --ckpt "$CK_ROOT/$ARM/ckpt_last.pt" --tree "$TREE" --latents "$LAT_DIR/LATENTS_GW_$ARM.npz" --states-pt "$LAT_DIR/latents/$ARM.pt" --out-dir "$OUT" > "$OUT/ctx_roll_$ARM.log" 2>&1
    RC=$?
    if [ "$RC" != "0" ]; then echo "ZZ CTXROLL STOP $ARM ctx_roll rc $RC"; tail -n 3 "$OUT/ctx_roll_$ARM.log"; exit 1; fi
  fi
  for V in R1 R3h R3g; do
    if [ ! -f "$OUT/rb/RESULT_RB_${ARM}_$V.json" ]; then
      "$PY" rb_trained.py run --arm "${ARM}_$V" --states "$ST_DIR/${ARM}_eval.npz" --latents "$OUT/LATENTS_CTX_${ARM}_$V.npz" --out-dir "$OUT/rb" --expect-dim 640 > "$OUT/rb/run_${ARM}_$V.log" 2>&1
      RC=$?
      if [ "$RC" != "0" ]; then echo "ZZ CTXROLL STOP $ARM rb $V rc $RC"; tail -n 3 "$OUT/rb/run_${ARM}_$V.log"; exit 1; fi
    fi
  done
  "$PY" rb_trained.py pair --a "$OUT/rb/OOF_RB_${ARM}_R3h.npz" --b "$OUT/rb/OOF_RB_${ARM}_R1.npz" --out "$OUT/rb/PAIR_RB_${ARM}_R3h_minus_R1.md" > /dev/null 2>&1 || { echo "ZZ CTXROLL STOP $ARM pair R3h R1"; exit 1; }
  "$PY" rb_trained.py pair --a "$OUT/rb/OOF_RB_${ARM}_R3g.npz" --b "$OUT/rb/OOF_RB_${ARM}_R1.npz" --out "$OUT/rb/PAIR_RB_${ARM}_R3g_minus_R1.md" > /dev/null 2>&1 || { echo "ZZ CTXROLL STOP $ARM pair R3g R1"; exit 1; }
  "$PY" rb_trained.py pair --a "$OUT/rb/OOF_RB_${ARM}_R3h.npz" --b "$OUT/rb/OOF_RB_${ARM}_R3g.npz" --out "$OUT/rb/PAIR_RB_${ARM}_R3h_minus_R3g.md" > /dev/null 2>&1 || { echo "ZZ CTXROLL STOP $ARM pair R3h R3g"; exit 1; }
  "$PY" rb_change.py run --oof "$OUT/rb/OOF_RB_${ARM}_R1.npz" --oof "$OUT/rb/OOF_RB_${ARM}_R3h.npz" --oof "$OUT/rb/OOF_RB_${ARM}_R3g.npz" --pair "${ARM}_R3h:${ARM}_R1" --pair "${ARM}_R3g:${ARM}_R1" --out "$OUT/rb/CHANGE_CTX_$ARM.md" > /dev/null 2>&1 || { echo "ZZ CTXROLL STOP $ARM change form"; exit 1; }
  cp "$OUT/RESULT_CTXROLL_$ARM.json" "$OUT/ctx_roll_$ARM.log" "$W/spatial/q2/ctx/" 2>/dev/null
  for V in R1 R3h R3g; do cp "$OUT/rb/RESULT_RB_${ARM}_$V.json" "$OUT/rb/VERDICT_RB_${ARM}_$V.md" "$W/spatial/q2/ctx/"; done
  cp "$OUT/rb/PAIR_RB_${ARM}_"*.md "$OUT/rb/CHANGE_CTX_$ARM.md" "$OUT/rb/CHANGE_CTX_$ARM.json" "$W/spatial/q2/ctx/"
  echo "ZZ CTXROLL ARM DONE $ARM $(date -u +%H:%M:%SZ)"
done
echo "ZZ CTXROLL ALL DONE"
