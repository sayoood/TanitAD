#!/usr/bin/env bash
# MM watch: one ACTSENS arm read (read_actsens.sh <tag>) ends with ARM_DONE / ARM_FAILED, or its driver disappears. usage: watch_read.sh <tag>
T=${1:?tag}
O=/f/TanitAD-scratch/actsens/read/out/$T
miss=0
for i in $(seq 1 180); do
  [ -f "$O/ARM_DONE_$T.txt" ] && { echo "READ $T DONE $(date '+%F %T')"; exit 0; }
  [ -f "$O/ARM_FAILED_$T.txt" ] && { echo "READ $T FAILED $(date '+%F %T') :: $(head -c 400 "$O/ARM_FAILED_$T.txt")"; exit 0; }
  pat="read_act""sens.sh $T"
  n="$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like ('*' + '$pat' + '*') }).Count" 2>/dev/null | tr -d '\r ')"
  if [ "$n" = "0" ]; then miss=$((miss+1)); else miss=0; fi
  [ "$miss" -ge 3 ] && { echo "READ $T DRIVER-GONE $(date '+%F %T') :: $(tail -c 300 "$O/read.log" | tr '\r\n' '  ')"; exit 0; }
  sleep 120
done
echo "READ $T WATCH-TIMEOUT $(date '+%F %T')"
