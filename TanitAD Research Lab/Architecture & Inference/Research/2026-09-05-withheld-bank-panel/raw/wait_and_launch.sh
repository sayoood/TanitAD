#!/bin/sh
# Wait for the sibling stream's refav1 eval to release the RTX 4060, then launch the
# panel. The GPU is shared and one arm at a time is the rule (torch spawns ~113
# threads per process; concurrent arms make no progress). Condition: no
# `refav1_arm.py` python process alive for 3 consecutive polls (90 s) AND GPU
# utilisation < 25 % on each of those polls. The condition is computed here and
# emitted as an opaque marker so a log grep can never match its own command.
W=/c/Users/Admin/run_wbank
quiet=0
while :; do
  n=$(powershell.exe -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'refav1_arm.py|refcv3_arm.py|t1_eval.py' }).Count" 2>/dev/null | tr -d '\r ')
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d '\r ')
  [ -z "$n" ] && n=99
  [ -z "$u" ] && u=100
  if [ "$n" -eq 0 ] && [ "$u" -lt 25 ]; then quiet=$((quiet+1)); else quiet=0; fi
  echo "ZZWAIT-$(date +%T)-procs${n}-util${u}-quiet${quiet}ZZ"
  [ "$quiet" -ge 3 ] && break
  sleep 30
done
echo "ZZLAUNCH-$(date +%T)ZZ"
sh "$W/panel.sh"
