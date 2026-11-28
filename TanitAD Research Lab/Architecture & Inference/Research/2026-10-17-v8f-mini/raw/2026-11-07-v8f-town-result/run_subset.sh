#!/usr/bin/env bash
# subset (first 10 episodes) controls at 4 threads: S_BARE = tool of record alone (dump-heads on), S_TOWN = town_run real twin. Compare in town_gates.py.
set -u
export PATH="/usr/bin:/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOWN_THREADS=4
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/town; PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/town
SW=F:/TanitAD-scratch/motion_c/town; mkdir -p /f/TanitAD-scratch/motion_c/town/out/receipts
ART=E:/Projects/TanitAD-artifacts; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TOOL=F:/TanitAD-scratch/motion_c/ri/tree_ri/stack/scripts/v8f_t1_eval_ri.py
COMMON=(--ckpt "$ART/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" --eval-join "$DATA/join_eval141.npz" --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --device cpu --smooth sg72 --scene reindexed)
G="$PKW/raw/grid_first10.json"
t0=$(date +%s)
"$PY" -I "$TOOL" run "${COMMON[@]}" --eval-grid "$G" --tag S_BARE --out-dir "$SW/out/S_BARE" --dump-ticks "$SW/out/S_BARE/ticks" --dump-heads < /dev/null > "$PK/logs/run_S_BARE.out" 2>&1; echo "bare rc=$? $(( $(date +%s)-t0 ))s"
t0=$(date +%s)
"$PY" -I "$PKW/town_run.py" --receipt-dir "$SW/out/receipts" -- "${COMMON[@]}" --eval-grid "$G" --tag S_TOWN --out-dir "$SW/out/S_TOWN" --dump-ticks "$SW/out/S_TOWN/ticks" --dump-heads < /dev/null > "$PK/logs/run_S_TOWN.out" 2>&1; echo "town rc=$? $(( $(date +%s)-t0 ))s"
echo done > "$PK/raw/SUBSET.done"
