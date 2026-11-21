#!/bin/bash
# Relay raw 256px-square val epcache episodes pod2 -> dev-box -> Thor.
# Streamed through a pipe (no local disk staging). -n on the SOURCE ssh so the
# nested ssh cannot eat the script's stdin (CLAUDE.md traps preflight).
SSH=/c/Windows/System32/OpenSSH/ssh.exe
SRC=/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
DST=/home/nvidia/valdata/physicalai-val-0c5f7dac3b11
N=${1:-5}
LOG=${RELAY_LOG:-./relay.log}
: > "$LOG"
$SSH -n -o BatchMode=yes -o StrictHostKeyChecking=no tanitad-thor "mkdir -p $DST" >>"$LOG" 2>&1
for i in $(seq 0 $((N-1))); do
  F=$(printf "ep_%05d.pt" "$i")
  # skip if already present with the right size
  SZ_SRC=$($SSH -n -o BatchMode=yes tanitad-pod2 "stat -c %s $SRC/$F" 2>/dev/null)
  SZ_DST=$($SSH -n -o BatchMode=yes tanitad-thor "stat -c %s $DST/$F 2>/dev/null || echo 0" 2>/dev/null)
  if [ "$SZ_SRC" = "$SZ_DST" ] && [ -n "$SZ_SRC" ]; then
    echo "[$(date -u +%H:%M:%S)] $F already present ($SZ_SRC B) - skip" >>"$LOG"; continue
  fi
  T0=$(date +%s)
  $SSH -n -o BatchMode=yes -o StrictHostKeyChecking=no tanitad-pod2 "cat $SRC/$F" \
    | $SSH -o BatchMode=yes -o StrictHostKeyChecking=no tanitad-thor "cat > $DST/$F"
  T1=$(date +%s)
  SZ_DST=$($SSH -n -o BatchMode=yes tanitad-thor "stat -c %s $DST/$F 2>/dev/null || echo 0" 2>/dev/null)
  DT=$((T1-T0)); [ "$DT" -eq 0 ] && DT=1
  echo "[$(date -u +%H:%M:%S)] $F src=$SZ_SRC dst=$SZ_DST ${DT}s $((SZ_DST/DT/1024/1024)) MB/s $([ "$SZ_SRC" = "$SZ_DST" ] && echo OK || echo SIZE_MISMATCH)" >>"$LOG"
done
# DONE marker mirrors the epcache contract; count reflects what was shipped
$SSH -n -o BatchMode=yes tanitad-thor "ls $DST/ep_*.pt | wc -l" >>"$LOG" 2>&1
echo "RELAY_COMPLETE" >>"$LOG"
