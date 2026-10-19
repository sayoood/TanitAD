#!/usr/bin/env bash
# Box 2 USB-HDD temperature guard for the RA-1 full RUNS (PREREG_RA1_FULL phases 1-3), 2026-10-19 06:41: temp_guard.sh covered only the
# build and exited with it, so the runs' 6-10 h of cache reads would have been unguarded. Same rule, unchanged:
#   STOP if >= 4 of the last 20 telemetry samples (15 s apart) read >= 61 C, or any one sample reads >= 63 C.
# Stop = Stop-Process on OUR job's python processes only (ra1_hjepa_train / ra1_hjepa_eval) and their workers. The chain then sees a
# nonzero rc and stops, and the v8F-mini and beta chains stop behind it (box 2 is left for a human). Exits when the runs chain ends.
B=/c/Users/Admin/box2_ship; L=$B/logs
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
echo "$(date '+%m-%d %H:%M') RA-1 runs temp guard armed (>=4/20 samples >= 61 C, or one >= 63 C)"
hi=0
while :; do
  grep -q -E 'ZZRA1FULLRUNS-DONE|ZZRA1FULLRUNS-STOP' $L/chain_ra1_full_runs.out 2>/dev/null && { echo "$(date '+%m-%d %H:%M') runs chain ended; guard off (max USB ${hi} C)"; exit 0; }
  t=$($SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 20\"" 2>/dev/null | tr -d '\r' | grep -E '^20' | cut -d, -f11 | grep -E '^[0-9]+$')
  if [ -n "$t" ]; then
    n61=$(echo "$t" | awk '$1 >= 61' | wc -l); m=$(echo "$t" | sort -n | tail -1)
    [ "$m" -gt "$hi" ] && { hi=$m; echo "$(date '+%m-%d %H:%M') new max USB temp $m C"; }
    if [ "$n61" -ge 4 ] || [ "$m" -ge 63 ]; then
      echo "ZZTEMP-STOP $(date '+%m-%d %H:%M') USB HDD: $n61 of the last 20 samples >= 61 C, max $m C -- stopping the RA-1 runs' python processes"
      $SSH "powershell -NoProfile -Command \"\$p = @(Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\"); \$ours = @(\$p | Where-Object { \$_.CommandLine -match 'ra1_hjepa_train|ra1_hjepa_eval' }); \$ids = @(\$ours | ForEach-Object { \$_.ProcessId }); \$kids = @(\$p | Where-Object { \$ids -contains \$_.ParentProcessId }); (\$ours + \$kids) | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue; 'stopped ' + \$_.ProcessId }\"" 2>/dev/null | tr -d '\r'
      exit 0
    fi
  fi
  sleep 60
done
