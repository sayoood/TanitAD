#!/usr/bin/env bash
# W1 ACCEPTANCE (SPEC.md §3) — the ONE command, from a CLEAN shell, when the box has RAM.
#
#   python -m taniteval.bench navsim_v2 --ckpt none --split warmup_two_stage --arms CV,STOP
#
# CLEAN SHELL: code/clean_shell.py builds the environment as an explicit dict (SystemRoot, PATH,
# TEMP, TMP, USERPROFILE, PYTHONPATH, PYTHONIOENCODING) — nothing NavSim-related is inherited, so
# the suite must set NUPLAN_* / NAVSIM_* / OPENSCENE_* / PYTHONHASHSEED itself or the acceptance is
# not clean. ⚠️ NOT `env -i` in MSYS bash: that hands a Windows child a POSIX-style PATH, which
# CreateProcess does not understand (MEASURED 2026-09-20: nvidia-smi was then unfindable).
#
# RAM: the promoted wrapper aborts (rc 3) below a 3,000 MB floor, and this box runs other agents'
# suites. MEASURED 2026-09-20 09:03: a 5,003 MB window collapsed to 1,176 MB within 20 s and the
# guard aborted — correctly. So: require >= ${RAM_MIN_MB} MB (default 6,000, the navhard chain's
# bar) and RETRY a retryable abort up to ${TRIES} times, 15 min apart.
# ⛔ A8's done-marker must exist (the heavy-compute rule).
set -u
PKG="D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-suite-core-navsim-v2"
LOG="$PKG/raw/acceptance_run.log"
A8_DONE="C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/summary.json"
RAM_MIN_MB="${RAM_MIN_MB:-6000}"
TRIES="${TRIES:-8}"
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"

avail_mb() {
  powershell.exe -NoProfile -Command "[int](Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes" 2>/dev/null | tr -cd '0-9'
}

wait_ram() {   # up to 3 h for a window
  local i a
  for i in $(seq 1 90); do
    a=$(avail_mb)
    if [ -n "$a" ] && [ "$a" -ge "$RAM_MIN_MB" ]; then echo "RAM window: ${a} MB available" >> "$LOG"; return 0; fi
    sleep 120
  done
  echo "NO RAM WINDOW in 3 h (last ${a:-?} MB)" >> "$LOG"
  return 1
}

echo "== W1 acceptance start $(date -u +%FT%TZ) (RAM_MIN_MB=$RAM_MIN_MB TRIES=$TRIES)" >> "$LOG"
if [ ! -f "$A8_DONE" ]; then echo "A8 done-marker absent — REFUSING heavy compute" >> "$LOG"; exit 5; fi
cd /d/Projects/TanitAD || exit 6

for k in $(seq 1 "$TRIES"); do
  wait_ram || { echo "attempt $k: no RAM window — stopping" >> "$LOG"; exit 4; }
  echo "-- attempt $k $(date -u +%FT%TZ)" >> "$LOG"
  "$PY" "$PKG/code/clean_shell.py" -- "$PY" -m taniteval.bench navsim_v2 --ckpt none --split warmup_two_stage --arms CV,STOP >> "$LOG" 2>&1
  rc=$?
  retryable=$(grep -a "^BENCH_RETRYABLE=" "$LOG" | tail -1 | cut -d= -f2)
  echo "attempt $k rc=$rc retryable=${retryable:-?} avail_after=$(avail_mb) $(date -u +%FT%TZ)" >> "$LOG"
  if [ "$rc" -eq 0 ]; then echo "ACCEPTANCE_RC=0 $(date -u +%FT%TZ)" >> "$LOG"; exit 0; fi
  if [ "${retryable:-0}" != "1" ]; then echo "ACCEPTANCE_RC=$rc (not retryable) $(date -u +%FT%TZ)" >> "$LOG"; exit "$rc"; fi
  sleep 900
done
echo "ACCEPTANCE_RC=3 (RAM guard aborted $TRIES attempts) $(date -u +%FT%TZ)" >> "$LOG"
exit 3
