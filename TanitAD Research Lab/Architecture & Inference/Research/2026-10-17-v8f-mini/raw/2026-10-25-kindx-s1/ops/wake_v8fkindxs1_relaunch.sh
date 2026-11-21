#!/usr/bin/env bash
# wake_v8fkindxs1_relaunch.sh: exits when the relauncher refuses / times out, the relaunched KINDX s1 chain writes ZZV8FMINI-DONE / -STOP,
# a guard stops it, or (after the relaunch) no chain_v8f_kindx_s1.sh driver is alive. Script polling, no model turns. Cap 7 h.
L=/c/Users/Admin/box2_ship/logs; O=$L/chain_v8f_kindx_s1.out; R=$L/relaunch_kindx_s1.out
alive() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name = 'bash.exe'\" | Where-Object { \$_.CommandLine -match 'chain_v8f_kindx_s1\.sh' }).Count" 2>/dev/null | tr -d '\r' | tail -1; }
for i in $(seq 1 210); do
  grep -aq 'ZZRELAUNCH-REFUSED\|ZZRELAUNCH-TIMEOUT' $R 2>/dev/null && { tail -4 $R; exit 2; }
  grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $O 2>/dev/null && { tail -15 $O; exit 0; }
  grep -aq ZZGUARD-STOP $L/v8fkindx_guard.out 2>/dev/null && { echo "GUARD: $(grep -a ZZGUARD-STOP $L/v8fkindx_guard.out | tail -1)"; exit 4; }
  if grep -aq ZZRELAUNCHED $R 2>/dev/null && [ "$i" -gt 1 ]; then
    a=$(alive); [ "$a" = 0 ] && { sleep 30; grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $O && { tail -15 $O; exit 0; }; echo "RELAUNCHED DRIVER GONE with no marker"; tail -8 $O; exit 3; }
  fi
  sleep 120
done
echo "WAKER TIMEOUT"; tail -5 $R; tail -5 $O
