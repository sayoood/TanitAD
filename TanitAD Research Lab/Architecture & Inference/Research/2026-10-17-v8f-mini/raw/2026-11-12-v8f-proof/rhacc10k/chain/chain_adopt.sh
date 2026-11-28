#!/usr/bin/env bash
# chain_adopt.sh <job> <match> <n0> -- adopts a box 2 job that is already running (launched by chain_rhacc.sh) when that chain ended early.
# Same rules as chain_rhacc.sh: takes C:/Users/Admin/box2_ship/BOX2_BUSY.txt atomically, restarts box 2's telemetry, polls every 60 s.
# Differences: the queue end is a NEW end marker (the count of QUEUE-PAUSED / QUEUE-COMPLETE lines in the job log above n0, the count
# present at adoption), and the temperature stop matches python processes whose command line contains <match> (not 'rhacc10k').
# Markers on stdout: ZZADOPT-END <state> <utc>. Script polling, not model turns.
set -u
JOB=${1:?job}; MATCH=${2:?match}; N0=${3:?n0}
B=/c/Users/Admin/box2_ship
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
LOG='D:\tanitad\logs\'"$JOB"'.log'
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
ps1() { $SSH "powershell -NoProfile -Command \"$1\"" 2>/dev/null | tr -d '\r'; }
end() {
  ps1 "New-Item -ItemType File -Force D:/tanitad/logs/telemetry.stop | Out-Null" > /dev/null
  [ -f $B/BOX2_BUSY.txt ] && mv $B/BOX2_BUSY.txt "$B/BOX2_BUSY.adopt_${JOB}_$1_$(date +%Y%m%d_%H%M%S).txt"
  echo "ZZADOPT-END $1 $(ts)"; exit 0
}
( set -C; echo "v8proof fork card ADD149 job $JOB (adopted) since $(ts). Do NOT launch box 2 jobs while this file exists." > $B/BOX2_BUSY.txt ) 2>/dev/null \
  || { echo "ZZADOPT-END busy-held $(ts) $(cat $B/BOX2_BUSY.txt)"; exit 1; }
boot0=$(ps1 "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('o')")
[ -n "$boot0" ] || end no-ssh
echo "$(ts) box 2 boot $boot0"
ps1 "if (Test-Path D:/tanitad/logs/telemetry.stop) { Remove-Item D:/tanitad/logs/telemetry.stop }" > /dev/null
o=$($SSH 'powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\launch.ps1 -Job D:\tanitad\jobs\telemetry_long.cmd -Log D:\tanitad\logs\telemetry_long.log' 2>/dev/null | tr -d '\r'); echo "$(ts) telemetry: $o"
cnt() { ps1 "@(Select-String -Path '$LOG' -Pattern 'QUEUE-PAUSED|QUEUE-COMPLETE').Count"; }
alive() { ps1 "@(Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\" | Where-Object { \$_.CommandLine -match '$MATCH' }).Count"; }
echo "$(ts) markers $(cnt) (n0 $N0), matching processes $(alive)"
hi=0; dead=0
while :; do
  sleep 60
  s=$(ps1 "if (Test-Path D:/tanitad/rhacc10k/STOP.txt) { 'Q-STOP' } else { 'Q-RUN' }")
  [ "$s" = "Q-STOP" ] && end queue-stop
  c=$(cnt); [[ "$c" =~ ^[0-9]+$ ]] && [ "$c" -gt "$N0" ] && end queue-end
  a=$(alive)
  if [[ "$a" =~ ^[0-9]+$ ]] && [ "$a" -eq 0 ]; then dead=$((dead+1)); [ $dead -ge 3 ] && end process-gone-no-marker; else dead=0; fi
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
