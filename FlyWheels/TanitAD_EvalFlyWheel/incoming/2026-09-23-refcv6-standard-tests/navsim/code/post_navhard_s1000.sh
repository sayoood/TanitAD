#!/usr/bin/env bash
# navhard post-processing, in two independent halves (each gated on its own ARTIFACT):
#   KH-nav  — once this package's navhard CV re-score exists: import W7's floors and compare the
#             re-score with W7's CV cell by cell + the official two-stage EPDMS to 1e-12 (SPEC §7);
#   s1000   — once the step-1,000 R6_A1 navhard score exists: statistics (+ the log-cluster
#             bootstrap), SPEC §5 ladder, four families on the 450 stage-1 scenes, tables.
#   bash code/post_navhard_s1000.sh kh      |   bash code/post_navhard_s1000.sh s1000
# Every output of the s1000 half is labelled PIPELINE-VALIDATION (SPEC §6).
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
INP=C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json
L="$P/raw/post_navhard_s1000.log"
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
what="${1:-s1000}"
echo "POST_${what}_START $(date -u +%FT%TZ)" >> "$L"
if [ "$what" = "kh" ]; then
  "$PY" "$P/code/import_floors.py" --split navhard_two_stage --check-cv "$P/raw/harness_repro_navhard" >> "$L" 2>&1
  "$PY" -c "
import json; p=json.load(open(r'$P/raw/floors/navhard_two_stage/FLOORS_PROVENANCE.json', encoding='utf-8'))
k=p.get('_KH_nav_check', {}); print('KH_NAV', k.get('verdict'), k.get('max_abs'), k.get('official_two_stage_EPDMS_a'), k.get('official_abs_diff'))" >> "$L" 2>&1
else
  SC="$P/raw/scores_navhard_s1000"
  "$PY" "$P/code/parse6.py" --split navhard_two_stage --scores "$SC" \
     --floors "$P/raw/floors/navhard_two_stage" --bridge "$P/raw/bridge_navhard_s1000" \
     --inputs "$INP" --label PIPELINE-VALIDATION --csv-suffix __navhard_two_stage \
     --out "$P/raw/summary_navhard_s1000.json" >> "$L" 2>&1
  "$PY" "$P/code/decompose6.py" --split navhard_two_stage --scores "$SC" \
     --floors "$P/raw/floors/navhard_two_stage" --bridge "$P/raw/bridge_navhard_s1000" \
     --inputs "$INP" --label PIPELINE-VALIDATION --csv-suffix __navhard_two_stage \
     --out "$P/raw/decomposition_navhard_s1000.json" >> "$L" 2>&1
  "$PY" "$P/code/families6.py" --seam "$P/raw/bridge_navhard_s1000/seam_R6_A1.npz" --inputs "$INP" \
     --stage 1 --label PIPELINE-VALIDATION --out "$P/raw/families_navhard_s1000.json" >> "$L" 2>&1
  if [ -f "$P/raw/summary_navhard_s1000.json" ]; then
    "$PY" "$P/code/report6.py" --navhard "$P/raw/summary_navhard_s1000.json" \
       --decomp "$P/raw/decomposition_navhard_s1000.json" \
       > "$P/raw/tables_navhard_s1000.txt" 2>> "$L"
  fi
  for f in raw/summary_navhard_s1000.json raw/decomposition_navhard_s1000.json \
           raw/families_navhard_s1000.json raw/tables_navhard_s1000.txt; do
    if [ -s "$P/$f" ]; then echo "ARTIFACT_OK $f" >> "$L"; else echo "ARTIFACT_MISSING $f" >> "$L"; fi
  done
fi
echo "ZZPOSTNAVHARD_${what}_DONEZZ $(date -u +%FT%TZ)" >> "$L"
