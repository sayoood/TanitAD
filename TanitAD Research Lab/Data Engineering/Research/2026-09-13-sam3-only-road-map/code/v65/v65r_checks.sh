#!/bin/bash
# Thor: LiDAR scores, PI checks and robust A2 for the offline-composed map v65r (compose.py: LINE_EDGE_GAP=1 LINE_MIN_LEN_M=1.0
# WLK_ISLAND_M2=3 EDGE_OBS=near XWALK=fft_tiles) against the delivered d0, one clip.
SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python
C8=$1; J=$2; LOG=$SM/v65r_checks.log
cd $SM/eval
for ARM in a f; do
  SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65r$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65r OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SRC > $SM/score_v65r${ARM}_$C8.log 2>&1
done
VARS="nobox65d0_all=$SM/render5_${C8}_v65d0a nobox65d0_front=$SM/render5_${C8}_v65d0f compose65r_all=$SM/render5_${C8}_v65ra compose65r_front=$SM/render5_${C8}_v65rf"
env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $SM/${C8}_v61s $SM/pi_checks_${C8}_r.json $J -5,40,-15,15 $VARS > $SM/pi_checks_${C8}_r.log 2>&1
env OMP_NUM_THREADS=4 $PY a2_robust2.py $C8 $SM/${C8}_v61s $VARS > $SM/a2_robust2_${C8}_r.log 2>&1
echo "ZZRCHK-$C8-scores$(ls $SM/sam3map_score_${C8}_v61s*_gt_v65r.json 2>/dev/null | wc -l)-checks$(grep -c ZZPICHECKS $SM/pi_checks_${C8}_r.log)-a2r$(grep -c ZZA2R $SM/a2_robust2_${C8}_r.log)ZZ" >> $LOG
