#!/usr/bin/env bash
# SMOOTH-T1 K1 MATCHED-THREAD rerun (READY, NOT RUN: the lab CPU cap is 4 threads; this is one ~3 min job at 16).  The banked DAG-OACC s0 rollout used torch's default of 16 threads on this box.
# K1OFF_M16_DAGOACC_s0 = the PATCHED tool, --smooth off, same arguments, 16 threads.  Compare with code/k1_diag_compare.py-style max abs deviation against the banked dump/ticks:
# if <= 1e-4 m (or bit-identical), K1 as written passes; the K2 and K5-floor clauses then need the four sg72 runs and the K1 run re-read at 16 threads the same way.
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/smooth_t1; S=/f/TanitAD-scratch/smooth_t1; OUT=$S/out_m16; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
tag=K1OFF_M16_DAGOACC_s0; mkdir -p "$OUT/$tag"
"$PY" -I "$S/tree_dag/stack/scripts/v8f_t1_eval.py" run --ckpt "$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt" --tag "$tag" \
  --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
  --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu --smooth off > "$PK/logs/$tag.out" 2>&1
echo "$(date +%FT%T) $tag rc=$?" | tee -a "$PK/logs/k1_matched16.out"
