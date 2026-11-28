#!/usr/bin/env bash
# KH — THE HARNESS STILL READS ITS KNOWN VALUES, through THIS package's own drivers (SPEC §7).
#   bash code/run_harness7.sh warmup_navtest   # warmup CV/STOP/ECHO vs E2, then navtest STOP vs W3
#   bash code/run_harness7.sh navhard          # navhard CV vs W7 (5,912 tokens, ~3 h CPU)
# One scorer at a time per queue, RAM-gated by score_queue7.sh (the box is shared). Every verdict
# is read from its JSON ARTIFACT, never from an exit code (CLAUDE.md: assert on the artifact).
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
E2RAW="$P/../../2026-09-19-navsim-refcv4b-bridge/raw"
W3RAW="$P/../../2026-09-19-navsim-v1-navtest/raw"
export PYTHONPATH="C:/Users/Admin/ev7nav/stack;C:/Users/Admin/ev7nav/taniteval"
what="${1:-warmup_navtest}"
L="$P/raw/harness_${what}.log"
echo "START $what $(date -u +%FT%TZ)" >> "$L"
verdict() {  # $1 = json, $2 = key path (dot) -> prints the value or ABSENT
  "$PY" -c "
import json,sys
try:
    d=json.load(open(r'$1',encoding='utf-8'))
    for k in '$2'.split('.'): d=d[k]
    print(d)
except Exception as e: print('ABSENT')"
}
if [ "$what" = "warmup_navtest" ]; then
  OUT="$P/raw/harness_repro"
  bash "$P/code/score_queue7.sh" warmup_two_stage "$OUT" e7kh \
       CV_official=OFFICIAL:constant_velocity_agent \
       STOP_zero="$E2RAW/seam_STOP_zero.npz" ECHO_ha0_ext="$E2RAW/seam_ECHO_ha0_ext.npz" >> "$L" 2>&1
  "$PY" "$P/code/check_harness_repro7.py" --mine-dir "$OUT" --arms CV_official,STOP_zero \
       --out "$P/raw/HARNESS_REPRO.json" >> "$L" 2>&1
  "$PY" "$P/code/check_harness_repro7.py" --mine-dir "$OUT" --arms ECHO_ha0_ext \
       --out "$P/raw/controls/KH_ECHO_warmup.json" >> "$L" 2>&1
  echo "KH_WARMUP CV_STOP=$(verdict "$P/raw/HARNESS_REPRO.json" verdict) ECHO=$(verdict "$P/raw/controls/KH_ECHO_warmup.json" verdict) $(date -u +%FT%TZ)" >> "$L"
  "$PY" "$P/code/import_floors7.py" --split warmup_two_stage >> "$L" 2>&1
  NT="$P/raw/harness_navtest"
  for try in 1 2 3 4 5 6; do
    c=$(verdict "$NT/r7kh_STOP/r7kh_STOP.counts.json" status)
    [ "$c" = "PASS" ] && break
    "$PY" -c "
import psutil,time
ok=0
while ok<3:
    ok = ok+1 if psutil.virtual_memory().available/2**30 >= 6 else 0
    time.sleep(0 if ok>=3 else 30)"
    "$PY" "$P/code/score_navtest7.py" --label r7kh_STOP --official STOP --out "$NT" >> "$NT.driver.txt" 2>&1
    echo "KH_NAVTEST_SCORE try=$try counts=$(verdict "$NT/r7kh_STOP/r7kh_STOP.counts.json" status) $(date -u +%FT%TZ)" >> "$L"
  done
  "$PY" "$P/code/check_kh_navtest7.py" --mine "$NT/r7kh_STOP/r7kh_STOP.csv" \
       --w3 "$W3RAW/STOP_navtest/STOP_navtest.csv" --out "$P/raw/controls/KH_navtest_STOP.json" >> "$L" 2>&1
  echo "KH_NAVTEST $(verdict "$P/raw/controls/KH_navtest_STOP.json" verdict) $(date -u +%FT%TZ)" >> "$L"
else
  OUT="$P/raw/harness_repro_navhard"
  RAM_MIN_GB=5 bash "$P/code/score_queue7.sh" navhard_two_stage "$OUT" e7kh \
       CV_official=OFFICIAL:constant_velocity_agent >> "$L" 2>&1
  "$PY" "$P/code/import_floors7.py" --split navhard_two_stage --check-cv "$OUT" >> "$L" 2>&1
  echo "KH_NAVHARD $(verdict "$P/raw/floors/navhard_two_stage/FLOORS_PROVENANCE.json" _KH_nav_check.verdict) $(date -u +%FT%TZ)" >> "$L"
fi
echo "ZZHARNESS_${what}_DONEZZ $(date -u +%FT%TZ)" >> "$L"
