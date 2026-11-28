#!/bin/bash
# mm_wait_read_s3.sh -- MM 2026-11-23. Waits for the dev-box SEED3 read (rh_read_s3_retry.sh, started via WMI at 20:54Z).
# Exit 0 = "ZZ S3-RETRY DONE ZZ"; 1 = "ZZ S3-RETRY STOP ZZ"; 4 = the retry driver is gone with no final line (checked every 10 min,
# two misses in a row); 3 = cap 8 h.
RL=F:/TanitAD-scratch/rollhist/read/out/rh_read_s3_retry.log
CL=F:/TanitAD-scratch/rollhist/read/out/rh_read_s3_chain.log
miss=0
for i in $(seq 1 480); do
  last=$(tail -n 1 "$RL" 2>/dev/null)
  case "$last" in
    *"ZZ S3-RETRY DONE ZZ"*) echo "READ DONE: $last"; tail -n 6 "$CL"; exit 0 ;;
    *"ZZ S3-RETRY STOP ZZ"*) echo "READ STOP: $last"; tail -n 12 "$CL"; exit 1 ;;
  esac
  if [ $((i % 10)) -eq 0 ]; then
    n=$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -like '*rh_read_s3_retry.sh*' }).Count" 2>/dev/null | tr -cd '0-9')
    if [ "${n:-0}" = 0 ]; then miss=$((miss + 1)); else miss=0; fi
    if [ $miss -ge 2 ]; then
      last=$(tail -n 1 "$RL" 2>/dev/null)
      case "$last" in *"ZZ S3-RETRY"*) continue ;; esac
      echo "READ DRIVER GONE (no bash.exe running rh_read_s3_retry.sh, twice), last retry line: $last"; tail -n 12 "$CL"; exit 4
    fi
  fi
  sleep 60
done
echo "READ WAIT TIMEOUT 8 h"; tail -n 6 "$CL"; exit 3
