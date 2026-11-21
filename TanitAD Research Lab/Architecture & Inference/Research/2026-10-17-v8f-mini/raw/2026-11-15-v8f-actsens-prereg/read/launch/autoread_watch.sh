#!/usr/bin/env bash
# MM: for each tag in order, wait for its BANK.done (and no read running), launch read_<tag>.cmd via WMI, then watch the read to its end.
# Prints one line per launch and per read end; exits after the LAST tag's read ends, or on the first FAILED / DRIVER-GONE / refusal (so the MM is woken).
# usage: autoread_watch.sh <tag> [<tag> ...]   (a copy of autoread.sh's launch logic with a longer wait: 1,200 polls of 60 s per tag)
L=/c/Users/Admin/actsens_launch
for T in "$@"; do
  B="C:/Users/Admin/t2v8f_pulls/actsens_$T/BANK.done"
  CMD="F:\\TanitAD-scratch\\actsens\\read\\runs\\read_$T.cmd"
  [ -f "/f/TanitAD-scratch/actsens/read/runs/read_$T.cmd" ] || { echo "AUTOREAD $T REFUSED: no read_$T.cmd"; exit 3; }
  [ -f /c/Users/Admin/actsens_state/STOP.txt ] && { echo "AUTOREAD $T CHAIN-STOP :: $(head -c 300 /c/Users/Admin/actsens_state/STOP.txt)"; exit 0; }
  launched=0
  for i in $(seq 1 1200); do
    if [ -n "$(ls /f/TanitAD-scratch/actsens/read/out/*/read.log 2>/dev/null)" ] && [ "$(powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'bash.exe' -and \$_.CommandLine -like ('*read_act' + 'sens.sh*') }).Count" 2>/dev/null | tr -d '\r ')" != "0" ]; then sleep 60; continue; fi
    if [ -s "$B" ]; then
      if [ -f "/f/TanitAD-scratch/actsens/read/out/$T/ARM_DONE_$T.txt" ]; then echo "AUTOREAD $T already read"; launched=2; break; fi
      r="$(powershell -NoProfile -Command "\$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags = [uint32]0x08000200; ShowWindow = [uint16]0}; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = 'cmd.exe /c $CMD < NUL'; CurrentDirectory = 'F:\\TanitAD-scratch\\actsens\\read\\runs'; ProcessStartupInformation = \$si}; 'rc=' + \$r.ReturnValue + ' pid=' + \$r.ProcessId" 2>&1 | tr -d '\r')"
      echo "AUTOREAD $T LAUNCHED $(date '+%F %T') $r"; launched=1; break
    fi
    [ -f /c/Users/Admin/actsens_state/STOP.txt ] && { echo "AUTOREAD $T CHAIN-STOP :: $(head -c 300 /c/Users/Admin/actsens_state/STOP.txt)"; exit 0; }
    sleep 60
  done
  [ "$launched" = "0" ] && { echo "AUTOREAD $T TIMEOUT $(date '+%F %T')"; exit 0; }
  [ "$launched" = "2" ] && continue
  sleep 90
  res="$(bash $L/watch_read.sh "$T")"; echo "$res"
  case "$res" in *" DONE "*) ;; *) exit 0;; esac
done
echo "AUTOREAD-WATCH ALL DONE $(date '+%F %T')"
