#!/usr/bin/env bash
# MM: wait until the chain banks <tag> (BANK.done), then start its read detached via WMI (read_<tag>.cmd) and exit. usage: autoread.sh <tag>
T=${1:?tag}
B="C:/Users/Admin/t2v8f_pulls/actsens_$T/BANK.done"
CMD="F:\\TanitAD-scratch\\actsens\\read\\runs\\read_$T.cmd"
[ -f "/f/TanitAD-scratch/actsens/read/runs/read_$T.cmd" ] || { echo "AUTOREAD $T REFUSED: no read_$T.cmd"; exit 3; }
for i in $(seq 1 420); do
  if [ -n "$(ls /f/TanitAD-scratch/actsens/read/out/*/read.log 2>/dev/null)" ] && [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like ('*read_act' + 'sens.sh*') }).Count" 2>/dev/null | tr -d '\r ')" != "0" ]; then sleep 60; continue; fi   # one read at a time
  if [ -s "$B" ]; then
    if [ -f "/f/TanitAD-scratch/actsens/read/out/$T/ARM_DONE_$T.txt" ]; then echo "AUTOREAD $T already read"; exit 0; fi
    r="$(powershell -NoProfile -Command "\$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags = [uint32]0x08000200; ShowWindow = [uint16]0}; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = 'cmd.exe /c $CMD < NUL'; CurrentDirectory = 'F:\\TanitAD-scratch\\actsens\\read\\runs'; ProcessStartupInformation = \$si}; 'rc=' + \$r.ReturnValue + ' pid=' + \$r.ProcessId" 2>&1 | tr -d '\r')"
    echo "AUTOREAD $T LAUNCHED $(date '+%F %T') $r"; exit 0
  fi
  sleep 60
done
echo "AUTOREAD $T TIMEOUT $(date '+%F %T')"
