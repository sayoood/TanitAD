#!/bin/bash
# Waiter: polls every 5 min; launches phase2_l1full.sh in-place when free RAM >= 6 GB and C: >= 6 GB; gives up after 12 h. Status in phase2_l1full/WAITER_STATUS.txt
export PATH="/usr/bin:/bin:$PATH"
O=/f/TanitAD-scratch/hwm_plan/phase2_l1full
ST=$O/WAITER_STATUS.txt
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PSH=/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
T0=$(date +%s)
while true; do
  NOW=$(date +%s)
  if [ $((NOW - T0)) -gt 43200 ]; then echo "GAVE_UP $(date -Iseconds) after 12 h, floor never held" > $ST; exit 3; fi
  RAM=$($PSH -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" | tr -d '\r' | tr ',' '.')
  CFREE=$($PSH -NoProfile -Command "[math]::Round((Get-PSDrive C).Free/1GB,2)" | tr -d '\r' | tr ',' '.')
  echo "WAITING $(date -Iseconds) ram_free_gb=$RAM c_free_gb=$CFREE" > $ST
  OK=$($PY -c "print(1 if float('$RAM')>=6 and float('$CFREE')>=6 else 0)")
  if [ "$OK" = "1" ]; then
    W=1; if $PY -c "import sys; sys.exit(0 if float('$RAM')>=8 else 1)"; then W=2; fi
    echo "LAUNCHING $(date -Iseconds) ram_free_gb=$RAM c_free_gb=$CFREE workers=$W" > $ST
    export HWM_WORKERS=$W
    bash /f/TanitAD-scratch/hwm_plan/phase2_l1full/phase2_l1full.sh > $O/phase2_l1full.log 2>&1
    echo "JOB_EXITED $(date -Iseconds) workers=$W (see PHASE2_L1FULL.done)" > $ST
    exit 0
  fi
  sleep 300
done
