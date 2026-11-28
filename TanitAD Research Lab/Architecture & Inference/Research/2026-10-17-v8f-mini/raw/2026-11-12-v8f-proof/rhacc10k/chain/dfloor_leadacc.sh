#!/usr/bin/env bash
# dfloor_leadacc.sh -- disk refusal floor for card ADD153 amendment 2 on box 2 (coordinator 2026-11-28: D: measured at 1.77 GB free).
# Polls box 2's D: free space every 60 s while seq_leadacc_am2.sh runs. Below FLOOR_GB it writes D:/tanitad/rhacc10k/STOP.txt (chain_rhacc.sh then ends the job as
# queue-stop and the sequence stops) and stops this card's python processes only (command line matching D:\tanitad\leadacc). Writes nothing else; deletes nothing.
# Markers: ZZDFLOOR <state> <free GB> <utc>. Script polling, not model turns.
set -u
FLOOR_GB=1.0
SEQ_OUT=F:/TanitAD-scratch/v8proof/rhacc10k/chain/seq_leadacc_am2.out
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
lo=99
while :; do
  grep -q "ZZSEQ-END" $SEQ_OUT && { echo "ZZDFLOOR seq-ended min $lo $(ts)"; exit 0; }
  f=$($SSH "powershell -NoProfile -Command \"[math]::Round((Get-PSDrive D).Free/1GB,3)\"" 2>/dev/null | tr -d '\r' | tr ',' '.' | grep -E '^[0-9.]+$')
  if [ -n "$f" ]; then
    awk -v a="$f" -v b="$lo" 'BEGIN{exit !(a<b)}' && { lo=$f; echo "$(ts) new min D: free $f GB"; }
    if awk -v a="$f" -v b="$FLOOR_GB" 'BEGIN{exit !(a<b)}'; then
      $SSH "powershell -NoProfile -Command \"Set-Content -Path D:/tanitad/rhacc10k/STOP.txt -Value 'ADD153 AM2 disk floor: D: free $f GB below $FLOOR_GB GB'; Get-CimInstance Win32_Process -Filter \\\"Name = 'python.exe'\\\" | Where-Object { \$_.CommandLine -match 'tanitad.leadacc' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force; 'stopped ' + \$_.ProcessId }\"" 2>/dev/null | tr -d '\r'
      echo "ZZDFLOOR below-floor $f $(ts)"; exit 0
    fi
  fi
  sleep 60
done
