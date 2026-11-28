#!/usr/bin/env bash
# waitjob.sh <logname> <max_min> <boot0>: poll a box 2 job log for ZZJOB-END; flag a reboot (crash). Script polling, not model turns.
L=/c/Users/Admin/box2_ship/logs; mkdir -p $L
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
miss=0
for i in $(seq 1 $2); do
  sleep 30
  b=$($SSH "powershell -NoProfile -Command \"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToString('yyyy-MM-dd HH:mm:ss')\"" 2>/dev/null | tr -d '\r' | grep -E '^[0-9]{4}-' | head -1)
  if [ -z "$b" ]; then miss=$((miss+1)); [ $miss -ge 24 ] && { echo "ZZUNREACHABLE 12 min"; exit 3; }; continue; fi
  miss=0; [ "$b" != "$3" ] && { echo "ZZCRASH box 2 rebooted: $3 -> $b"; exit 2; }
  scp -q -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2:D:/tanitad/logs/$1.log $L/ 2>/dev/null
  grep -aq 'ZZJOB-END' $L/$1.log 2>/dev/null && { tr '\r' '\n' < $L/$1.log | grep -a 'ZZJOB-END' | tail -1; exit 0; }
done
echo "ZZTIMEOUT $2 polls"; exit 4
