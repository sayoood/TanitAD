#!/usr/bin/env bash
# After the corrected step-1,000 warmup queue (code/run_warmup_s1000.sh) finishes: the THIRD harness
# control (ECHO re-scored vs E2's banked ECHO, same rule as KH), the floors, the statistics, the SPEC §5
# ladder, and the markdown tables — every output labelled PIPELINE-VALIDATION (SPEC §6).
# Each step is gated on the ARTIFACT of the one before; nothing here reads an exit code as evidence.
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
E2RAW="$P/../../2026-09-19-navsim-refcv4b-bridge/raw"
SC="$P/raw/scores_warmup_s1000"
L="$P/raw/post_warmup_s1000.log"
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
echo "POST_START $(date -u +%FT%TZ)" >> "$L"
"$PY" "$P/code/check_harness_repro.py" --mine-dir "$SC" --arms ECHO_ha0_ext \
   --out "$P/raw/controls/KH_ECHO_warmup.json" >> "$L" 2>&1
"$PY" "$P/code/import_floors.py" --split warmup_two_stage >> "$L" 2>&1
"$PY" "$P/code/parse6.py" --split warmup_two_stage --scores "$SC" \
   --floors "$P/raw/floors/warmup_two_stage" --bridge "$P/raw/bridge_warmup_s1000" \
   --inputs "$E2RAW/navsim_agent_inputs.json" --label PIPELINE-VALIDATION \
   --out "$P/raw/summary_warmup_s1000.json" >> "$L" 2>&1
"$PY" "$P/code/decompose6.py" --split warmup_two_stage --scores "$SC" \
   --floors "$P/raw/floors/warmup_two_stage" --bridge "$P/raw/bridge_warmup_s1000" \
   --inputs "$E2RAW/navsim_agent_inputs.json" --label PIPELINE-VALIDATION \
   --out "$P/raw/decomposition_warmup_s1000.json" >> "$L" 2>&1
"$PY" "$P/code/plan_deltas.py" --bridge "$P/raw/bridge_warmup_s1000" --label PIPELINE-VALIDATION \
   --out "$P/raw/controls/plan_deltas_warmup_s1000.json" >> "$L" 2>&1
if [ -f "$P/raw/summary_warmup_s1000.json" ]; then
  "$PY" "$P/code/report6.py" --warmup "$P/raw/summary_warmup_s1000.json" \
     --plans "$P/raw/controls/plan_deltas_warmup_s1000.json" \
     --decomp "$P/raw/decomposition_warmup_s1000.json" \
     > "$P/raw/tables_warmup_s1000.txt" 2>> "$L"
fi
for f in raw/controls/KH_ECHO_warmup.json raw/floors/warmup_two_stage/FLOORS_PROVENANCE.json \
         raw/summary_warmup_s1000.json raw/decomposition_warmup_s1000.json raw/tables_warmup_s1000.txt; do
  if [ -s "$P/$f" ]; then echo "ARTIFACT_OK $f" >> "$L"; else echo "ARTIFACT_MISSING $f" >> "$L"; fi
done
echo "ZZPOSTWARMUPDONEZZ $(date -u +%FT%TZ)" >> "$L"
