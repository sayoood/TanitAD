#!/usr/bin/env bash
# Resume the navhard reference chain ONLY AFTER A8 finishes (Master Mind request 2026-09-19: no load on
# a box that is training; A8 had slipped 2.51 -> ~7.6 s/step beside the first cache attempt, which was
# stopped by explicit PIDs). Trigger: A8's done-marker `run/summary.json`. Then: ONE worker, all NavSim
# output on C: (run_navhard_c1.sh), >= 6,000 MB AVAILABLE before each attempt, RAM-abort (rc 3)
# retried up to 8x 15 min apart. Steps: cache -> N1 (CV, official two-stage) -> N2b (human, stage 1).
set -u
A8_DONE="C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/summary.json"
RUN="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navhard-download/code/run_navhard_c1.sh"
RAW="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navhard-download/raw"
LOG="$RAW/navhard_reference_chain.log"

avail_mb() {
  powershell.exe -NoProfile -Command "[int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" 2>/dev/null | tr -cd '0-9'
}

wait_ram() {
  local i a
  for i in $(seq 1 180); do
    a=$(avail_mb)
    if [ -n "$a" ] && [ "$a" -ge 6000 ]; then return 0; fi
    sleep 60
  done
  return 1
}

run_step() {  # $1 step, $2 max attempts
  local step=$1 n=$2 k rc t0
  for k in $(seq 1 "$n"); do
    if ! wait_ram; then
      echo "step=$step attempt=$k NO_RAM_WINDOW_3h $(date -u +%FT%TZ)" >> "$LOG"
      return 4
    fi
    t0=$(date +%s)
    bash "$RUN" "$step" > "$RAW/chain_c1_${step}_try${k}.log" 2>&1
    rc=$?
    echo "c1 step=$step attempt=$k rc=$rc avail_mb_after=$(avail_mb) wall_s=$(( $(date +%s) - t0 )) end=$(date -u +%FT%TZ)" >> "$LOG"
    if [ "$rc" -eq 0 ]; then return 0; fi
    if [ "$rc" -ne 3 ]; then return "$rc"; fi
    sleep 900
  done
  return 3
}

echo "after-A8 waiter start $(date -u +%FT%TZ) (stopped attempt: 7 PIDs 2864/18964/27732/34108/45456/43672/52952 by explicit Stop-Process)" >> "$LOG"
for i in $(seq 1 180); do   # wait up to 15 h for A8's done-marker, polling every 5 min
  [ -f "$A8_DONE" ] && break
  sleep 300
done
if [ ! -f "$A8_DONE" ]; then echo "A8 done-marker never appeared in 15 h - abort, nothing started" >> "$LOG"; exit 5; fi
echo "A8 done-marker seen $(date -u +%FT%TZ); starting C:-only 1-worker chain" >> "$LOG"
for s in cache N1 N2b; do
  run_step "$s" 8 || { echo "STOP at $s" >> "$LOG"; exit 1; }
done
echo "CHAIN_DONE $(date -u +%FT%TZ)" >> "$LOG"
