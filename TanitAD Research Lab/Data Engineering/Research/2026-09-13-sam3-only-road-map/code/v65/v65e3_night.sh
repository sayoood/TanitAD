#!/bin/bash
# Thor: pre-registered arm e3 (renderer v5f docstring) on the night clip: e2 (BG_VOTE_NEAR=10) + BG_EDGES=1 + BG_MINCOMP=50,
# on top of v6.5d, ALL 7 and FRONT; LiDAR scores; PI checks; robust A2 (on-road).
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
LOG=$SM/v65e3_night.log
C8=73495082f98b
cd $SM/eval
for ARM in a f; do
  SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 AGENT_DRIVABLE=1 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 OMP_NUM_THREADS=4 \
      BG_VOTE_NEAR=10 BG_EDGES=1 BG_MINCOMP=50 $PY sam3map_render_v5f.py $C8 $SRC $SM/render5_${C8}_v65e3$ARM > $SM/render5_${C8}_v65e3$ARM.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65e3$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65e3 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SRC > $SM/score_v65e3${ARM}_$C8.log 2>&1
done
S=$SM
VARS="paint65d_all=$S/render5_${C8}_v65da bgvote65e2_all=$S/render5_${C8}_v65e2a bgedge65e3_all=$S/render5_${C8}_v65e3a bgedge65e3_front=$S/render5_${C8}_v65e3f"
env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $S/${C8}_v61s $S/pi_checks_${C8}_d.json 86 -5,40,-15,15 $VARS > $S/pi_checks_${C8}_d.log 2>&1
env OMP_NUM_THREADS=4 $PY a2_robust2.py $C8 $S/${C8}_v61s $VARS > $S/a2_robust2_${C8}_d.log 2>&1
NM=$(ls $SM/render5_${C8}_v65e3[af]/worldmap.npz 2>/dev/null | wc -l)
echo "ZZV65E3N-maps$NM-checks$(grep -c ZZPICHECKS $S/pi_checks_${C8}_d.log)-a2r$(grep -c ZZA2R $S/a2_robust2_${C8}_d.log)ZZ" >> $LOG
