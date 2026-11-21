#!/usr/bin/env bash
# W3's ONE waiter: sleep silently, wake the agent ONLY on a substantive event.
#
# ⭐ WHY (orchestrator, 2026-09-20): every wake costs budget, and three consecutive progress pings
# carried nothing but a percentage and an ETA band that were already known. A long quiet stretch
# during a 4-hour cache is the EXPECTED state, not a risk — so this exits (which is the single
# notification) only when something has actually happened:
#
#   DONE_FILE_PRESENT  the awaited artifact appeared (arg 4; default the analysis JSON)
#   STEP_NONZERO       the chain logged a step with rc != 0 SINCE ITS LAST START
#   CHAIN_GONE         no after_a8_chain process, and no analysis file  -> it died
#   LOW_RAM_<n>_for_<k>_checks   available memory stayed below the floor for k checks (not a dip)
#   TIMEOUT_<h>H       nothing happened in the whole window (also worth one wake)
#
# ⚠️ Two traps avoided on purpose:
#   * the process probe prints a COUNT from PowerShell, never a grep of a command line that
#     contains the pattern being searched for (`pgrep -f` family: the filter matching its own echo);
#   * rc lines are read only AFTER the last "chain start", because the log still carries the rc=1
#     of the pass the RAM guard aborted at 06:52 — an old failure is not a new one.
set -u
PKG="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest"
LOG="$PKG/raw/chain_w3/chain.log"
CACHE="D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest"
# the artifact whose APPEARANCE means "the thing I am waiting for is done". It moves as the chain
# moves: raw/analysis_navtest.json for the reproduction, then the bank's BANK_DONE.json.
DONE_FILE=${4:-"$PKG/raw/analysis_navtest.json"}
MAX_ITER=${1:-210}          # x 2 min = 7 h
RAM_FLOOR_MB=${2:-2500}
RAM_SUSTAIN=${5:-3}         # consecutive sub-floor checks (x 10 min apart) before it counts
low=0
# ⛔ rc lines are counted only from BASE onward: the log already holds the rc=1 of the step the RAM
# guard aborted, and an old failure must not fire a new wake (it did, once — that is why this arg
# exists). Default: everything already written is history.
BASE=${3:-$(wc -l < "$LOG" 2>/dev/null || echo 0)}
reason=""
i=0
while [ "$i" -lt "$MAX_ITER" ]; do
  [ -f "$DONE_FILE" ] && { reason="DONE_FILE_PRESENT"; break; }
  if [ -f "$LOG" ] && tail -n "+$((BASE + 1))" "$LOG" | grep -qE "rc=[1-9]"; then
    reason="STEP_NONZERO"; break
  fi
  if [ $((i % 5)) -eq 0 ]; then
    alive=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -and \$_.CommandLine -like '*after_a8*' } | Measure-Object).Count" 2>/dev/null | tr -cd '0-9')
    [ "${alive:-1}" = "0" ] && { reason="CHAIN_GONE"; break; }
    avail=$(powershell.exe -NoProfile -Command "[int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" 2>/dev/null | tr -cd '0-9')
    # ⚠️ SUSTAIN, never a single sample. MEASURED 2026-09-20: one dip to 1,900 MB woke the agent
    # for a condition the chain already handles by waiting, and memory was back to 3,413 MB
    # seconds later. This is E1's own lesson (its RAM guard v1 killed a healthy 194 MB import on
    # a transient dip caused by OTHER processes) applied to the waiter.
    if [ -n "${avail:-}" ] && [ "$avail" -lt "$RAM_FLOOR_MB" ]; then
      low=$((low + 1))
      [ "$low" -ge "$RAM_SUSTAIN" ] && { reason="LOW_RAM_${avail}_for_${low}_checks"; break; }
    else
      low=0
    fi
  fi
  sleep 120
  i=$((i + 1))
done
pkl=$(find "$CACHE" -name metric_cache.pkl 2>/dev/null | wc -l)
csvs=$(find "$PKG/raw" -name "*_navtest*.csv" 2>/dev/null | wc -l)
bank=$(ls -1 "$(dirname "$DONE_FILE")" 2>/dev/null | wc -l)
shards=$(ls -1 D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/frame_bank/shard_*.DONE.json 2>/dev/null | wc -l)
echo "W3_WAITER_FIRED reason=${reason:-TIMEOUT} waited_min=$((i * 2)) pkl=${pkl}/12146 navtest_csvs=${csvs} bank_shards_done=${shards}/32 done_file=${DONE_FILE} utc=$(date -u +%FT%TZ)"
tail -3 "$LOG" 2>/dev/null | cut -c1-200
