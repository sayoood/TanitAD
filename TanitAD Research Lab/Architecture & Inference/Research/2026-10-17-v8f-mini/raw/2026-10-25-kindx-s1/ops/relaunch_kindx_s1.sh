#!/usr/bin/env bash
# relaunch_kindx_s1.sh (MM, 2026-10-25): the KINDX seed-1 chain's driver died at ~19:45 (cygwin fork failures under commit pressure) while
# its trainer kept running on box 2. This waits for that trainer to END, then restarts the chain, which skips the finished run
# (chain_v8f_kindx_s1.sh: "box 2 already holds a finished M2b_s1_mixo: trainmixo skips it") and does the evals, controls and read.
# Never touches a running trainer. Stops (no relaunch) on a guard stop. Script polling, no model turns.
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH"
B=/c/Users/Admin/box2_ship; L=$B/logs; LOG=$L/relaunch_kindx_s1.out
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
JL='D:/tanitad/logs/v8fkindx10251607_t_M2b_s1_mixo.log'; RUN='D:/tanitad/runs/v8f_mini_full/M2b_s1_mixo'
say() { echo "$(date '+%m-%d %H:%M') $*" >> $LOG; }
say "armed: waiting for the trainer's ZZJOB-END in $JL and $RUN/summary.json"
for i in $(seq 1 150); do   # 150 x 120 s = 5 h
  grep -aq ZZGUARD-STOP $L/v8fkindx_guard.out 2>/dev/null && { say "ZZRELAUNCH-REFUSED a guard stop is in v8fkindx_guard.out; not relaunching"; exit 4; }
  r=$($SSH "powershell -NoProfile -Command \"if ((Select-String -Path $JL -Pattern 'ZZJOB-END' -Quiet) -and (Test-Path $RUN/summary.json)) { 'ENDED' } else { 'RUNNING' }\"" 2>/dev/null | tr -d '\r' | tail -1)
  [ "$r" = ENDED ] && break
  sleep 120
done
[ "$r" = ENDED ] || { say "ZZRELAUNCH-TIMEOUT the trainer did not end within 5 h (last: '$r')"; exit 5; }
say "the trainer ended: $($SSH "powershell -NoProfile -Command \"Select-String -Path $JL -Pattern 'ZZJOB-END' | Select-Object -Last 1 | ForEach-Object { \$_.Line }\"" 2>/dev/null | tr -d '\r' | tail -1)"
n=$($SSH "powershell -NoProfile -Command \"@(Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\" | Where-Object { \$_.CommandLine -match 'v8f_mini_train' }).Count\"" 2>/dev/null | tr -d '\r' | tail -1)
[ "$n" = 0 ] || { say "ZZRELAUNCH-REFUSED box 2 still runs $n v8f_mini_train process(es)"; exit 6; }
[ -f $B/BOX2_BUSY.txt ] && { mv $B/BOX2_BUSY.txt $L/BOX2_BUSY.stale_kindxs1_driver_died_1945.txt && say "moved the dead driver's busy marker aside"; }
[ -f $L/chain_v8f_kindx_s1.out ] && { mv $L/chain_v8f_kindx_s1.out $L/chain_v8f_kindx_s1.driver_died_1945.out && say "moved the dead driver's log aside"; }
cmd.exe //c "C:\\Users\\Admin\\box2_ship\\run_chain_v8f_kindx_s1.cmd < NUL" &
say "ZZRELAUNCHED the KINDX seed-1 chain (log logs/chain_v8f_kindx_s1.out)"
wait
say "the relaunched chain's wrapper exited: $(tail -1 $L/chain_v8f_kindx_s1.out 2>/dev/null | cut -c1-160)"
