#!/bin/bash
# Thor: checks for rule rs2 (v65rs2_preregistration.txt): crosswalk image contrast, PI checks, LiDAR scores, robust A2; then videos.
SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python; LOG=$SM/v65rs2_checks.log
for PAIR in 4fbd97b6a4b7:41 73495082f98b:86; do
  C8=${PAIR%%:*}; J=${PAIR##*:}
  cd $SM/eval
  env OMP_NUM_THREADS=4 nice -n 5 $PY xwalk_reproj_contrast.py $C8 $SM/${C8}_v61s $SM/render5_${C8}_v65ma "r delivered=$SM/render5_${C8}_v65ra" "rs2 soft stripes=$SM/render5_${C8}_v65rs2a" > $SM/xrc_rs2_$C8.log 2>&1
  VARS="compose65r_all=$SM/render5_${C8}_v65ra compose65r_front=$SM/render5_${C8}_v65rf soft65rs2_all=$SM/render5_${C8}_v65rs2a soft65rs2_front=$SM/render5_${C8}_v65rs2f"
  env OMP_NUM_THREADS=4 nice -n 5 $PY pi_checks.py $C8 $SM/${C8}_v61s $SM/pi_checks_${C8}_rs2.json $J -5,40,-15,15 $VARS > $SM/pi_checks_${C8}_rs2.log 2>&1
  for ARM in a f; do
    SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65rs2$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65rs2 OMP_NUM_THREADS=4 nice -n 5 $PY thor_run.py score $C8 $SRC > $SM/score_v65rs2${ARM}_$C8.log 2>&1
  done
  env OMP_NUM_THREADS=4 nice -n 5 $PY a2_robust2.py $C8 $SM/${C8}_v61s $VARS > $SM/a2_robust2_${C8}_rs2.log 2>&1
  echo "ZZRS2-$C8-xrc$(grep -c ZZXRC $SM/xrc_rs2_$C8.log)-checks$(grep -c ZZPICHECKS $SM/pi_checks_${C8}_rs2.log)-a2r$(grep -c ZZA2R $SM/a2_robust2_${C8}_rs2.log)ZZ" >> $LOG
done
