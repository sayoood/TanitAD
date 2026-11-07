#!/usr/bin/env bash
# subset (TOWN's first 10 episodes / 170 windows) at 4 threads: S_OWN (arm own), S_TSWAP, S_MUT; gates vs TOWN's S_BARE (the bare pinned tool, 4 threads). Outputs: F:/TanitAD-scratch/motion_c/tswap/out/S_*, raw/SUB_*.json
set -u
export PATH="/usr/bin:/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TSWAP_THREADS=4
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/tswap; PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/tswap
SW=F:/TanitAD-scratch/motion_c/tswap; BARE=F:/TanitAD-scratch/motion_c/town/out
ART=E:/Projects/TanitAD-artifacts; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
COMMON=(--ckpt "$ART/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" --eval-join "$DATA/join_eval141.npz" --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --device cpu --smooth sg72 --scene reindexed)
G="$PKW/raw/grid_first10.json"
mkdir -p /f/TanitAD-scratch/motion_c/tswap/out/receipts
for spec in "own S_OWN" "tswap S_TSWAP" "mutant S_MUT"; do
  set -- $spec; arm=$1; tag=$2; rm -rf "/f/TanitAD-scratch/motion_c/tswap/out/$tag"; t0=$(date +%s)
  "$PY" -I "$PKW/tswap_run.py" --arm "$arm" --receipt-dir "$SW/out/receipts" -- "${COMMON[@]}" --eval-grid "$G" --tag "$tag" --out-dir "$SW/out/$tag" --dump-ticks "$SW/out/$tag/ticks" --dump-heads < /dev/null > "$PK/logs/run_$tag.out" 2>&1
  echo "$tag rc=$? $(( $(date +%s)-t0 ))s"
done
"$PY" -I "$PKW/tswap_gates.py" --own "$SW/out/S_OWN" --own-tag S_OWN --bare "$BARE/S_BARE" --bare-tag S_BARE --tswap "$SW/out/S_TSWAP" --tswap-tag S_TSWAP --mutant "$SW/out/S_MUT" --mutant-tag S_MUT \
  --rec-tswap "$SW/out/receipts/receipt_S_TSWAP.json" --rec-mutant "$SW/out/receipts/receipt_S_MUT.json" --out "$PKW/raw/SUB_GATES.json" > "$PK/logs/sub_gates.out" 2>&1
echo "gates rc=$?"
