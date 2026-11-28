#!/usr/bin/env bash
# Box 2 USB-HDD temperature guard for the RA-1 full build (2026-10-18 21:50): the bus-powered Seagate (33k power-on hours) climbed
# from a 54 to a 59 C hourly peak over 9 h of build. Portable 2.5" drives are typically rated to 60 C. Rule (operational, fixed now):
#   STOP the build if >= 4 of the last 20 samples (15 s apart = 5 min) read >= 61 C, or any one sample reads >= 63 C.
# Stop = Stop-Process on OUR job's python processes only (ra1_full_box2 / pod_build_b1_epcache / pull_encode_pooled / verify_cache).
# The build resumes per clip (builder and encoder both skip finished clips; ra1_full_state.json keeps finished chunks).
# Exits when the build chain ends. Script polling, not model turns.
B=/c/Users/Admin/box2_ship; L=$B/logs
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
echo "$(date '+%m-%d %H:%M') temp guard armed (>=4/20 samples >= 61 C, or one >= 63 C)"
hi=0
while :; do
  grep -q -E 'ZZRA1FULL-BUILD-DONE|ZZRA1FULL-STOP' $L/chain_ra1_full.out 2>/dev/null && { echo "$(date '+%m-%d %H:%M') build chain ended; guard off (hourly max seen ${hi} C)"; exit 0; }
  t=$($SSH "powershell -NoProfile -Command \"Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 20\"" 2>/dev/null | tr -d '\r' | grep -E '^20' | cut -d, -f11 | grep -E '^[0-9]+$')
  if [ -n "$t" ]; then
    n61=$(echo "$t" | awk '$1 >= 61' | wc -l); m=$(echo "$t" | sort -n | tail -1)
    [ "$m" -gt "$hi" ] && { hi=$m; echo "$(date '+%m-%d %H:%M') new max USB temp $m C"; }
    if [ "$n61" -ge 4 ] || [ "$m" -ge 63 ]; then
      echo "ZZTEMP-STOP $(date '+%m-%d %H:%M') USB HDD: $n61 of the last 20 samples >= 61 C, max $m C -- stopping the build's python processes"
      $SSH "powershell -NoProfile -Command \"\$p = @(Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\"); \$ours = @(\$p | Where-Object { \$_.CommandLine -match 'ra1_full_box2|pod_build_b1_epcache|pull_encode_pooled|verify_cache' }); \$ids = @(\$ours | ForEach-Object { \$_.ProcessId }); \$kids = @(\$p | Where-Object { \$ids -contains \$_.ParentProcessId }); (\$ours + \$kids) | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue; 'stopped ' + \$_.ProcessId }\"" 2>/dev/null | tr -d '\r'
      exit 0
    fi
  fi
  sleep 60
done
