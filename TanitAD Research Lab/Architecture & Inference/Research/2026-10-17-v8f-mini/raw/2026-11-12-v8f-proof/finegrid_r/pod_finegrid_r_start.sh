#!/usr/bin/env bash
# pod_finegrid_r_start.sh : runs ON THE POD under nohup, started by the lead. Card PREREG_V8F_PROOF_ADD43 with note N1.
# Order of the pod: the run of card ADD43 takes its first slot only after the launcher of note ADD37_N7 has started its second chain
# (marker STARTED_actlevel2_1.txt) or has refused (ACTLEVEL2_NOT_STARTED.txt), so two slot waits never race for the same free slot.
# At most 10 h, then it goes on (the run script's own slot wait still refuses a fourth GPU job). No process test, no rm, no mv.
Z=/workspace/v8proof/runs/actlevel2_z
j=0
while [ ! -e $Z/STARTED_actlevel2_1.txt ] && [ ! -e $Z/ACTLEVEL2_NOT_STARTED.txt ]; do
  j=$((j+1)); [ $j -gt 600 ] && break
  sleep 60
done
echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] finegrid_r start: order wait ended after $j min ($(ls $Z | grep -c 'STARTED_actlevel2_1\|NOT_STARTED') marker)"
cd /workspace/v8proof/finegrid || exit 1
exec bash pod_finegrid_r_run.sh
