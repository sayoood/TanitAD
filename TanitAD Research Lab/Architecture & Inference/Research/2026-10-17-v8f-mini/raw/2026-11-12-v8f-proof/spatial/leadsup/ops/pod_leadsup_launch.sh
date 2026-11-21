#!/usr/bin/env bash
# pod_leadsup_launch.sh : runs ON THE POD (bash -s). Starts the three gated chains of card PREREG_V8F_PROOF_ADD24 (two arms each, one arm at a time per chain).
# leadsup_a waits for the DONE marker of spatialrep_a; leadsup_b and leadsup_c wait for the DONE marker of spatialrep_b: never more than three trainers.
set -u
W=/workspace/v8proof; CH=$W/chains; cd $W/chain2 || exit 1
echo "trainer processes before launch: $(pgrep -f v8f_mini_train.py | wc -l)"
anon=$(grep -E '^anon ' /sys/fs/cgroup/memory.stat | cut -d' ' -f2); max=$(cat /sys/fs/cgroup/memory.max)
free=$(( (max - anon) / 1073741824 )); echo "container memory free GB: $free"
[ "$free" -ge 30 ] || { echo "REFUSED: less than 30 GB free"; exit 1; }
for c in spatialrep_a spatialrep_b; do ls $CH/$c/ | grep -q STOP && { echo "REFUSED: $c has a STOP marker"; exit 1; }; done
start() { local c=$1 pred=$2
  [ -e $CH/$c/chain.log ] && { echo "REFUSED: $c exists"; return 1; }
  mkdir -p $CH/$c
  C2_IDLE_MIB=30000 C2_WAIT_FOR=$CH/$pred/${pred}_DONE.txt C2_CARD=PREREG_V8F_PROOF_ADD24 nohup setsid bash chain_arms.sh $c arms_$c.txt > $CH/$c.nohup 2>&1 < /dev/null &
  sleep 5; echo "started $c (waits for ${pred}_DONE.txt)"; }
start leadsup_a spatialrep_a
start leadsup_b spatialrep_b
start leadsup_c spatialrep_b
sleep 20
for c in leadsup_a leadsup_b leadsup_c; do echo "== $c"; ls $CH/$c | tr '\n' ' '; echo; tail -n 3 $CH/$c/chain.log 2>/dev/null | cut -c1-220; tail -n 2 $CH/$c.nohup | cut -c1-200; done
pgrep -f "chain_arms.sh leadsup" | wc -l
for c in spatialrep_a spatialrep_b; do tail -n 1 $CH/$c/chain.log | cut -c1-160; done
tail -n 1 $W/runs/spatialrep_a/G_s3/train_log.jsonl | cut -c1-40
