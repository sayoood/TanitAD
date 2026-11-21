#!/usr/bin/env bash
# MM watch: C7b (read package on R35_s0, MKL 1 fix) ends with ARM_DONE / ARM_FAILED, or its driver disappears.
O=/f/TanitAD-scratch/actsens/read/c7diag/out/C7b_R35_s0
T=C7b_R35_s0
miss=0
for i in $(seq 1 120); do
  [ -f "$O/ARM_DONE_$T.txt" ] && { echo "C7b DONE $(date '+%F %T')"; exit 0; }
  [ -f "$O/ARM_FAILED_$T.txt" ] && { echo "C7b FAILED $(date '+%F %T') :: $(head -c 400 "$O/ARM_FAILED_$T.txt")"; exit 0; }
  pat="read_act""sens.sh $T"
  n="$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like ('*' + '$pat' + '*') }).Count" 2>/dev/null | tr -d '\r ')"
  if [ "$n" = "0" ]; then miss=$((miss+1)); else miss=0; fi
  [ "$miss" -ge 3 ] && { echo "C7b DRIVER-GONE $(date '+%F %T') :: $(tail -c 300 "$O/read.log" | tr '\r\n' '  ')"; exit 0; }
  sleep 120
done
echo "C7b WATCH-TIMEOUT $(date '+%F %T')"
