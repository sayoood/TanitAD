#!/usr/bin/env bash
# ⭐ FIT-NOISE CONTROL. The trajectory is five SINGLE-SEED probe fits, so a
# between-point difference has no error bar of its own. This re-fits the SAME
# @11250 cache at two more seeds: the spread across seeds is the yardstick the
# trajectory's spread must be judged against. Cheap — sp2 never loads the trunk.
set -u
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=4
until [ -f "$S/cache_s11250/latents.pt" ]; do sleep 20; done
for sd in 1 2; do
  "$PY" "$S/sp2_probe.py" --cache "$S/cache_s11250/latents.pt" \
      --out "$S/out_s11250_seed$sd" --split-json "$S/p3_selection.json" \
      --arms cells --n-queries 74 --steps 3000 --batch 32 --lr 1e-3 --seed $sd
  cp "$S/out_s11250_seed$sd/slot_probe_results.json" "$S/raw/results_s11250_seed$sd.json"
  echo "SEEDDONE $sd"
done
echo "SEEDVAR_DONE"
