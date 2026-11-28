#!/usr/bin/env bash
# chain_rhacc.sh <job: rhacc_q1 | rhacc_qall> -- dev-box chain of card ADD139 on box 2 (one job at a time).
# Takes C:/Users/Admin/box2_ship/BOX2_BUSY.txt atomically, starts box 2's telemetry, checks it is fresh, launches the queue job
# detached (launch.ps1), then polls every 60 s: queue end (QUEUE-PAUSED / QUEUE-COMPLETE in its log, or D:/tanitad/rhacc10k/STOP.txt),
# a box 2 reboot (LastBootUpTime changes), and the USB-disk temperature rule of temp_guard.sh (>= 4 of the last 20 samples >= 61 C,
# or one >= 63 C: stops this card's python processes only). At the end: telemetry stopped, busy marker renamed.
# Markers on stdout (this chain's own log): ZZRHACC-END <state> <utc>. Script polling, not model turns.
set -u
JOB=${1:?job}
MATCH=${2:-rhacc10k}   # python command lines the temperature rule stops (card ADD149: a job outside rhacc10k passes its own)
B=/c/Users/Admin/box2_ship
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
LOG='D:\tanitad\logs\'"$JOB"'.log'
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
ps1() { $SSH "powershell -NoProfile -Command \"$1\"" 2>/dev/null | tr -d '\r'; }
end() {
  ps1 "New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null" > /dev/null
  [ -f $B/BOX2_BUSY.txt ] && mv $B/BOX2_BUSY.txt "$B/BOX2_BUSY.rhacc_${JOB}_$1_$(date +%Y%m%d_%H%M%S).txt"
  echo "ZZRHACC-END $1 $(ts)"; exit 0
}
( set -C; echo "v8proof fork card ADD139 RHACC10K job $JOB since $(ts). Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null \
  || { echo "ZZRHACC-END busy-held $(ts) $(cat $B/BOX2_BUSY.txt)"; exit 1; }
boot0=$(ps1 "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('o')")
[ -n "$boot0" ] || end no-ssh
echo "$(ts) box 2 boot $boot0"
ps1 "if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }" > /dev/null
o=$($SSH 'powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\telemetry_long.cmd -Log D:\tanitad\logs\telemetry_long.log' 2>/dev/null | tr -d '\r'); echo "$(ts) telemetry: $o"
tl() { ps1 "Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 1"; }
sleep 45; t1=$(tl); sleep 35; t2=$(tl)
[[ $t2 == 20* && $t2 != "$t1" ]] || end no-fresh-telemetry
echo "$(ts) telemetry fresh"
cnt() { ps1 "if (Test-Path '$LOG') { @(Select-String -Path '$LOG' -Pattern 'QUEUE-PAUSED|QUEUE-COMPLETE').Count } else { 0 }"; }
N0=$(cnt); [[ "$N0" =~ ^[0-9]+$ ]] || end no-marker-count
echo "$(ts) end markers already in the job log: $N0 (the queue ends on a new one; card ADD149)"
o=$($SSH "powershell -NoProfile -ExecutionPolicy Bypass -File D:\\tanitad\\launch.ps1 -Job D:\\tanitad\\jobs\\$JOB.cmd -Log $LOG" 2>/dev/null | tr -d '\r'); echo "$(ts) queue: $o"
[[ $o == *"rc=0"* ]] || end launch-failed
hi=0
while :; do
  sleep 60
  q=$(ps1 "if (Test-Path D:/tanitad/rhacc10k/STOP.txt) { 'Q-STOP' } else { 'Q-RUN' }")
  [ "$q" = "Q-STOP" ] && end queue-stop
  c=$(cnt); [[ "$c" =~ ^[0-9]+$ ]] && [ "$c" -gt "$N0" ] && end queue-end
  b=$(ps1 "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('o')")
  [ -n "$b" ] && [ "$b" != "$boot0" ] && end box2-rebooted
  t=$(ps1 "Get-Content D:/tanitad/logs/telemetry_full.csv -Tail 20" | grep -E '^20' | cut -d, -f11 | grep -E '^[0-9]+$')
  if [ -n "$t" ]; then
    n61=$(echo "$t" | awk '$1 >= 61' | wc -l); m=$(echo "$t" | sort -n | tail -1)
    [ "$m" -gt "$hi" ] && { hi=$m; echo "$(ts) new max USB temp $m C"; }
    if [ "$n61" -ge 4 ] || [ "$m" -ge 63 ]; then
      ps1 "Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\" | Where-Object { \$_.CommandLine -match '$MATCH' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force; 'stopped ' + \$_.ProcessId }"
      end temperature
    fi
  fi
done
