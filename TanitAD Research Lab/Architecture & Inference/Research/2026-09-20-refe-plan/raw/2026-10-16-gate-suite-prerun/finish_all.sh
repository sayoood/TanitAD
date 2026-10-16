#!/bin/bash
# The long half of the G-SUITE pre-run package, run DETACHED (WMI) so it survives the agent that started it.  Stages (status lines in logs/finish.status):
#   A  the 13 suite scripts of every profile, on the CHANGED tree  -> baseline/scripts_after/        (the BEFORE half is baseline/scripts_before/)
#   B  the new test in full (arms + 26-mutation matrix) + the arm table -> logs/test_gate_suite_prerun_full.log, arm_report.json     (A and B run side by side)
#   C  the real-profile measurement (suite-prerun on refe_p2rep, then run --only G-SUITE with and without the evidence) -> measure_p2rep/
#   D  tables/*.txt from the artifacts (make_tables.py)
# Marker: DONE_GATE_SUITE_PRERUN (all four stages ran; read tables/ALL.txt and logs/finish.status for what each said) or FAILED_GATE_SUITE_PRERUN.
R="$(cd "$(dirname "$0")" && pwd)"
PK="$(cd "$R/../.." && pwd)"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONUTF8=1 TMP="C:/Users/Admin/AppData/Local/Temp" TEMP="C:/Users/Admin/AppData/Local/Temp"
unset TANITAD_GATE_ARM
mkdir -p "$R/logs"
S="$R/logs/finish.status"
st() { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
rm -f "$R/DONE_GATE_SUITE_PRERUN" "$R/FAILED_GATE_SUITE_PRERUN"; : > "$S"
st "START pid $$"
rm -rf "$R/baseline/scripts_after" "$R/measure_p2rep"
PY=$PY "$R/baseline/run_suite_scripts.sh" "$R/baseline/scripts_after" > "$R/logs/stageA.out" 2>&1 &
PA=$!
( cd "$PK/refe" && "$PY" test_gate_suite_prerun.py --arm-report "$R/arm_report.json" ) > "$R/logs/test_gate_suite_prerun_full.log" 2>&1 &
PB=$!
wait $PB; RB=$?; st "B new test rc=$RB: $(grep -E 'ZZTEST_PASS|Error' "$R/logs/test_gate_suite_prerun_full.log" | tail -n 2 | tr '\n' ' ')"
wait $PA; RA=$?; st "A suite scripts rc=$RA ($(wc -l < "$R/baseline/scripts_after/summary.txt") lines in summary)"
"$PY" "$R/measure_real_profile.py" --pkg "$PK" --out "$R/measure_p2rep" > "$R/logs/measure.log" 2>&1; RC=$?; st "C measure rc=$RC"
"$PY" "$R/make_tables.py" > "$R/logs/make_tables.log" 2>&1; RD=$?; st "D tables rc=$RD: $(tail -n 1 "$R/logs/make_tables.log")"
if [ $RB -eq 0 ] && [ $RC -eq 0 ] && [ $RD -eq 0 ]; then touch "$R/DONE_GATE_SUITE_PRERUN"; st "DONE"; else touch "$R/FAILED_GATE_SUITE_PRERUN"; st "FAILED (B=$RB C=$RC D=$RD; A's rc is per script in scripts_after/summary.txt)"; fi
