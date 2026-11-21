#!/bin/bash
# Thor: score a clip's v1 and v2 SAM3 maps with the calibrated LiDAR checks (optionally after its v2 chain finishes).
C8=$1
SM=/home/nvidia/sam3map
if [ "$2" = "wait" ]; then
  gone=0
  while ! grep -qs "^ZZV2" $SM/render_${C8}_v2.render.log; do
    if [ "$(ps -eo args | grep -c "[r]ender_v2.sh ${C8}")" -eq 0 ]; then
      gone=$((gone + 1)); [ $gone -ge 2 ] && { echo "v2 chain ${C8} gone without its marker" >> $SM/score_${C8}_v2.log; break; }
    else
      gone=0
    fi
    sleep 30
  done
fi
cd $SM/eval || exit 1
for d in ${C8} ${C8}_v2; do
  OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py score ${C8} $SM/$d > $SM/score_$d.log 2>&1
  echo "score_exit=$?" >> $SM/score_$d.log
done
echo "ZZSCORED-${C8}ZZ" >> $SM/score_${C8}_v2.log
