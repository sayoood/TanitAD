#!/usr/bin/env bash
# Retry wrapper for the navhard reference chain after E1's RAM guard aborted `cache` (rc 3) at
# 2026-09-19T12:07Z: system AVAILABLE memory 2,627 MB < the guard's 3,000 MB floor. Competing jobs
# at the time were not TanitAD eval work (an earnings-swing sweep, several ffmpeg encodes, plus the
# A8 training run at ~5.8 GB). The guard is RIGHT and stays on. This waits for >= 6,000 MB
# AVAILABLE before each attempt, retries a RAM abort (rc 3) up to 8 times 15 min apart, then runs
# N1 (CV, official two-stage) and N2b (human, stage 1 only).
# AVAILABLE = free + standby (Win32_PerfFormattedData_PerfOS_Memory.AvailableMBytes), the quantity
# E1's guard reads. FreePhysicalMemory excludes standby and reads ~10x lower.
set -u
RUN="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/code/run_navhard.sh"
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
    bash "$RUN" "$step" > "$RAW/chain_${step}_try${k}.log" 2>&1
    rc=$?
    echo "step=$step attempt=$k rc=$rc avail_mb_after=$(avail_mb) wall_s=$(( $(date +%s) - t0 )) end=$(date -u +%FT%TZ)" >> "$LOG"
    if [ "$rc" -eq 0 ]; then return 0; fi
    if [ "$rc" -ne 3 ]; then return "$rc"; fi
    sleep 900
  done
  return 3
}

echo "retry chain start $(date -u +%FT%TZ) avail_mb=$(avail_mb)" >> "$LOG"
for s in cache N1 N2b; do
  run_step "$s" 8 || { echo "STOP at $s" >> "$LOG"; exit 1; }
done
echo "CHAIN_DONE $(date -u +%FT%TZ)" >> "$LOG"
