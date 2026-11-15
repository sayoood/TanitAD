#!/usr/bin/env bash
# MM watch: C7 (read package on R35_s0) ends with ARM_DONE_C7_R35_s0.txt or ARM_FAILED_C7_R35_s0.txt, or its driver disappears.
O=/f/TanitAD-scratch/actsens/read/out/C7_R35_s0
miss=0
for i in $(seq 1 180); do
  [ -f "$O/ARM_DONE_C7_R35_s0.txt" ] && { echo "C7 DONE $(date '+%F %T')"; exit 0; }
  [ -f "$O/ARM_FAILED_C7_R35_s0.txt" ] && { echo "C7 FAILED $(date '+%F %T') :: $(head -c 400 "$O/ARM_FAILED_C7_R35_s0.txt")"; exit 0; }
  pat="c7_r""un.sh"
  n="$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like ('*' + '$pat' + '*') }).Count" 2>/dev/null | tr -d '\r ')"
  if [ "$n" = "0" ]; then miss=$((miss+1)); else miss=0; fi
  [ "$miss" -ge 3 ] && { echo "C7 DRIVER-GONE $(date '+%F %T') :: tail $(tail -c 300 "$O/read.log" | tr '\r\n' '  ')"; exit 0; }
  sleep 120
done
echo "C7 WATCH-TIMEOUT $(date '+%F %T')"
