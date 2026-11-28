#!/bin/bash
# mm_wait_read_ki.sh -- MM 2026-11-23. Waits for the dev-box KI read (rh_read_ki_retry.sh, started via WMI at 13:45Z).
# Exit 0 = "ZZ KI-RETRY DONE ZZ"; 1 = "ZZ KI-RETRY STOP ZZ"; 4 = the retry driver is gone with no final line (checked every 10 min,
# two misses in a row); 3 = cap 8 h.
RL=F:/TanitAD-scratch/rollhist/read/out/rh_read_ki_retry.log
CL=F:/TanitAD-scratch/rollhist/read/out/rh_read_ki_chain.log
miss=0
for i in $(seq 1 480); do
  last=$(tail -n 1 "$RL" 2>/dev/null)
  case "$last" in
    *"ZZ KI-RETRY DONE ZZ"*) echo "READ DONE: $last"; tail -n 6 "$CL"; exit 0 ;;
    *"ZZ KI-RETRY STOP ZZ"*) echo "READ STOP: $last"; tail -n 12 "$CL"; exit 1 ;;
  esac
  if [ $((i % 10)) -eq 0 ]; then
    n=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*rh_read_ki_retry.sh*' }).Count" 2>/dev/null | tr -cd '0-9')
    if [ "${n:-0}" = 0 ]; then miss=$((miss + 1)); else miss=0; fi
    if [ $miss -ge 2 ]; then
      last=$(tail -n 1 "$RL" 2>/dev/null)
      case "$last" in *"ZZ KI-RETRY"*) continue ;; esac
      echo "READ DRIVER GONE (no bash.exe running rh_read_ki_retry.sh, twice), last retry line: $last"; tail -n 12 "$CL"; exit 4
    fi
  fi
  sleep 60
done
echo "READ WAIT TIMEOUT 8 h"; tail -n 6 "$CL"; exit 3
