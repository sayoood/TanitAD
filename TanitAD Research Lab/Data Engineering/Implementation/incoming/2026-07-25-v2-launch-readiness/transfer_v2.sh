#!/bin/bash
# v2 shard consolidation: tanitad-pod3 -> tanitad-pod (streamed, resumable, size-verified)
SP="/c/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
SSH="C:/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
SRC=/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d
STAGE=/workspace/data/physicalai_v2/_incoming_pod3
LOG="$SP/transfer.log"
LANES=4
echo "=== transfer start $(date -u +%FT%TZ) ===" >> "$LOG"
for round in $(seq 1 15); do
  $SSH tanitad-pod "cd $STAGE 2>/dev/null && stat -c '%n %s' *.v2ep.pt 2>/dev/null" > "$SP/staged.txt" 2>/dev/null
  awk 'NR==FNR{have[$1]=$2; next} { if (have[$1] != $2) print $1 }' "$SP/staged.txt" "$SP/v2_pod3_list.txt" > "$SP/delta.txt"
  n=$(wc -l < "$SP/delta.txt"); done_n=$(wc -l < "$SP/staged.txt")
  bytes=$(awk '{s+=$2} END{print s+0}' "$SP/staged.txt")
  echo "[$(date -u +%FT%TZ)] round=$round staged=$done_n remaining=$n bytes=$bytes ($(awk -v b=$bytes 'BEGIN{printf "%.2f", b/1073741824}') GiB)" >> "$LOG"
  if [ "$n" -eq 0 ]; then echo "ALL_TRANSFERRED $(date -u +%FT%TZ)" >> "$LOG"; break; fi
  rm -f "$SP"/lane_[0-9]
  awk -v L=$LANES -v P="$SP/lane_" '{f = P (NR % L); print >> f}' "$SP/delta.txt"
  for i in 0 1 2 3; do
    [ -s "$SP/lane_$i" ] || continue
    (
      $SSH tanitad-pod3 "cat > /workspace/tmp/lane_$i.txt" < "$SP/lane_$i"
      $SSH tanitad-pod3 "cd $SRC && tar -cf - --files-from=/workspace/tmp/lane_$i.txt" 2>/dev/null | \
        $SSH tanitad-pod "cd $STAGE && tar --no-same-owner -xf -" 2>/dev/null
      echo "  [round $round] lane $i finished rc=$? $(date -u +%FT%TZ)" >> "$LOG"
    ) &
  done
  wait
done
echo "=== driver exit $(date -u +%FT%TZ) ===" >> "$LOG"
