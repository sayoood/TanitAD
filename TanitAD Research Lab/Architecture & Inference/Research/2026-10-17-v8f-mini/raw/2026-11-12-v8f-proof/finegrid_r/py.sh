#!/bin/bash
# dev-box python wrapper: waits for >= 5.5 GB free commit memory (60 s steps), then runs the venv python on CPU. Usage: bash py.sh <args of python>
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1
n=0
while true; do
  f=$(powershell -NoProfile -Command '(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory' | tr -d '\r\n ')
  [ -n "$f" ] && [ "$f" -ge 5767168 ] && break
  n=$((n+1)); [ "$n" -gt 120 ] && { echo "py.sh: free commit memory stayed below 5.5 GB for 2 h"; exit 9; }
  sleep 60
done
exec /c/Users/Admin/venvs/tanitad/Scripts/python.exe "$@"
