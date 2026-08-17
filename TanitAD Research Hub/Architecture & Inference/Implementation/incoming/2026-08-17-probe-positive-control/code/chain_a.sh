#!/usr/bin/env bash
# CHAIN A — the positive control, 3 seeds, on the GT-oracle cache.
# ⛔ sp2_probe.py is BYTE-IDENTICAL to the parity run's (md5 aabbee36…): same
# fit, same controls, same estimator, same windows, same split.
set -u
W="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/pc"
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=6
CACHE="${1:-$W/cache_orc010/latents.pt}"
TAG="${2:-orc010}"
SEEDS="${3:-0 1 2}"
for sd in $SEEDS; do
  L="${TAG}_seed${sd}"
  if [ -f "$W/raw/results_$L.json" ]; then echo "PCSKIP $L"; continue; fi
  "$PY" "$W/sp2_probe.py" --cache "$CACHE" --out "$W/out_$L" \
      --split-json "$S/p3_selection.json" --arms cells \
      --n-queries 74 --steps 3000 --batch 32 --lr 1e-3 --seed "$sd" \
      > "$W/log_$L.txt" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "PCFAIL $L rc=$rc"; tail -20 "$W/log_$L.txt"; continue; fi
  cp "$W/out_$L/slot_probe_results.json" "$W/raw/results_$L.json"
  echo "PCDONE $L"
done
echo "CHAIN_A_DONE $TAG"
