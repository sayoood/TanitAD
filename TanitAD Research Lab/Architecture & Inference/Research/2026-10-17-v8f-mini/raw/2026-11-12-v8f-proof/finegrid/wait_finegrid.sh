#!/bin/bash
# wait_finegrid.sh -- dev box. Polls the pod every 300 s for the end marker of the pilot of card PREREG_V8F_PROOF_ADD41 (runs/finegrid/FINEGRID_DONE.txt or
# FINEGRID_STOP.txt). On an end it pulls the small text files of the run (the marker, the logs, G2_GRIDS.json, the train logs) into finegrid/in_pod/ and ends.
# The large state files stay on the pod (the dev box has no room for them); the reader is run there by the lead. Starts nothing on the pod. Deletes nothing.
# Last line: ZZ FINEGRID-POD DONE | STOP | TIMEOUT.
set -u
cd /f/TanitAD-scratch/v8proof/finegrid || exit 9
P=/c/Users/Admin/v8pod/pssh.sh
C=/c/Users/Admin/v8pod/pcat.sh
R=/workspace/v8proof/runs/finegrid
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
pod() { timeout 60 bash "$P" "$1" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
n=0; st=""
while true; do
  s=$(pod "cd $R 2>/dev/null && { [ -f FINEGRID_DONE.txt ] && echo DONE; [ -f FINEGRID_STOP.txt ] && echo STOP; echo MARK; tail -n 1 finegrid.log 2>/dev/null | cut -c1-150; tail -n 1 finegrid_start.log 2>/dev/null | cut -c1-110; }")
  case "$s" in DONE*) st=DONE; break ;; STOP*) st=STOP; break ;; esac
  n=$((n+1))
  [ $((n % 6)) = 1 ] && echo "[$(ts)] poll $n: $(echo "$s" | tr '\n' ';' | cut -c1-280)"
  if [ "$n" -gt 216 ]; then st=TIMEOUT; break; fi
  sleep 300
done
echo "[$(ts)] pilot state $st"
mkdir -p in_pod
if [ "$st" != TIMEOUT ]; then
  for f in FINEGRID_DONE.txt FINEGRID_STOP.txt finegrid.log finegrid_start.log G2_GRIDS.json train/TRAIN_DONE.json train/train_meta.json p3/TRAIN_DONE.json p3/train_meta.json \
           train/F16_s0/train_log.jsonl train/F16_s1/train_log.jsonl train/F64_s0/train_log.jsonl train/F64_s1/train_log.jsonl p3/F16SH_s0/train_log.jsonl; do
    d=in_pod/$(dirname "$f"); mkdir -p "$d"
    a=$(pod "sha256sum $R/$f 2>/dev/null | cut -d' ' -f1")
    [ -n "$a" ] || { echo "[$(ts)] not on the pod: $f"; continue; }
    bash "$C" "$R/$f" "in_pod/$f" < /dev/null > /dev/null 2>&1
    b=$(sha256sum "in_pod/$f" 2>/dev/null | cut -d' ' -f1)
    if [ "$a" = "$b" ]; then echo "[$(ts)] pulled $f sha256 ${a:0:16}"; else echo "[$(ts)] PULL-MISMATCH $f"; fi
  done
  pod "ls -la /workspace/v8proof/finegrid/states 2>/dev/null | awk '{print \$5, \$9}' | tail -n 30" > in_pod/states_listing.txt
  tail -n 4 in_pod/finegrid.log 2>/dev/null | cut -c1-240
fi
echo "ZZ FINEGRID-POD $st $(ts)"
