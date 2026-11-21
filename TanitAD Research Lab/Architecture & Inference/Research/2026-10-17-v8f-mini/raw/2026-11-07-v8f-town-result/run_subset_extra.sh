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
G="$PKW/raw/grid_first10.json"
mk() { local tag=$1 th=$2; shift 2; OMP_NUM_THREADS=$th MKL_NUM_THREADS=$th TOWN_THREADS=$th "$@" ; }
t0=$(date +%s); mk 4 4 "$PY" -I "$PKW/town_run.py" --twin-mode mutant --receipt-dir "$SW/out/receipts" -- "${COMMON[@]}" --eval-grid "$G" --tag S_MUT --out-dir "$SW/out/S_MUT" --dump-ticks "$SW/out/S_MUT/ticks" --dump-heads < /dev/null > "$PK/logs/run_S_MUT.out" 2>&1; echo "mut rc=$? $(( $(date +%s)-t0 ))s"
t0=$(date +%s); mk 4 4 "$PY" -I "$PKW/town_run.py" --zero-t-ego --receipt-dir "$SW/out/receipts" -- "${COMMON[@]}" --eval-grid "$G" --tag S_REG --out-dir "$SW/out/S_REG" --dump-ticks "$SW/out/S_REG/ticks" --dump-heads < /dev/null > "$PK/logs/run_S_REG.out" 2>&1; echo "reg rc=$? $(( $(date +%s)-t0 ))s"
t0=$(date +%s); mk 2 2 "$PY" -I "$TOOL" run "${COMMON[@]}" --eval-grid "$G" --tag S_BARE2 --out-dir "$SW/out/S_BARE2" --dump-ticks "$SW/out/S_BARE2/ticks" --dump-heads < /dev/null > "$PK/logs/run_S_BARE2.out" 2>&1; echo "bare2threads rc=$? $(( $(date +%s)-t0 ))s"
echo done > "$PK/raw/SUBSET_EXTRA.done"
bash "$PK/chain_town.sh"
