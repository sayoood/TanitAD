#!/usr/bin/env bash
# CHAIN C — ⭐ IS THE APPARATUS REPAIRABLE BY OPERATING POINT?
# The 74-query head puts ~8 slots in the 3.5 m-wide corridor BY CHANCE, and the
# incumbent rule then argmaxes presence over that clutter. This runs the SAME
# probe on the SAME oracle caches at SMALLER n_slot_queries. Pure fits — no
# trunk compute. If the oracle PASSES K1 at a small n_queries, the apparatus is
# repairable and we know how; if it fails even at ORC-DIRECT + 8 queries, it is
# not repairable by operating point.
set -u
W="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/pc"
S="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONUTF8=1 OMP_NUM_THREADS=4
run () {   # <tag> <cache> <nq>
  [ -f "$W/raw/results_$1.json" ] && { echo "PCSKIP $1"; return; }
  "$PY" "$W/sp2_probe.py" --cache "$2" --out "$W/out_$1" \
      --split-json "$S/p3_selection.json" --arms cells \
      --n-queries "$3" --steps 3000 --batch 32 --lr 1e-3 --seed 0 \
      > "$W/log_$1.txt" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "PCFAIL $1 rc=$rc"; tail -6 "$W/log_$1.txt"; return; fi
  cp "$W/out_$1/slot_probe_results.json" "$W/raw/results_$1.json"
  echo "PCDONE $1"
}
run orc010_nq16  "$W/cache_orc010/latents.pt"  16
run orcdir_nq8   "$W/cache_orcdir/latents.pt"   8
run orc010_nq8   "$W/cache_orc010/latents.pt"   8
echo "CHAIN_C_DONE"
# ⛔ ADDED 2026-08-17: THE REPAIR NEEDS ITS OWN NEGATIVE CONTROL. A rule/geometry
# change that makes the ORACLE pass K1 has fixed nothing if it also makes NOISE
# and the REAL ARM pass. That is the exact error class this whole package exists
# to catch, so the fix does not get a free ride.
run nullmatched_nq16 "$S/cache_nullmatched/latents.pt" 16
run s11250_nq16      "$S/cache_s11250/latents.pt"      16
echo "CHAIN_C2_DONE"
