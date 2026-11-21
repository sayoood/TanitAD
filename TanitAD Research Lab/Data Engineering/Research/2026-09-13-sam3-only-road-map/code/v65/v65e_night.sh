#!/bin/bash
# Thor: the two pre-registered boundary arms on the night clip (renderer v5e docstring): e1 MAJ_WEIGHT=sharp4, e2 BG_VOTE_NEAR=10,
# both on top of v6.5d, ALL 7 and FRONT; LiDAR scores; PI checks; robust A2 (on-road).
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
LOG=$SM/v65e_night.log
C8=73495082f98b
BASE="SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 AGENT_DRIVABLE=1 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 OMP_NUM_THREADS=4"
cd $SM/eval
for E in e1 e2; do
  LEVER=$([ $E = e1 ] && echo "MAJ_WEIGHT=sharp4" || echo "BG_VOTE_NEAR=10")
  for ARM in a f; do
    SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
    env $BASE $LEVER $PY sam3map_render_v5e.py $C8 $SRC $SM/render5_${C8}_v65${E}$ARM > $SM/render5_${C8}_v65${E}$ARM.log 2>&1
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65${E}$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65$E OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SRC > $SM/score_v65${E}${ARM}_$C8.log 2>&1
  done
done
S=$SM
VARS="nearest_all=$S/render5_${C8}_v61 paint65d_all=$S/render5_${C8}_v65da paint65d_front=$S/render5_${C8}_v65df sharp65e1_all=$S/render5_${C8}_v65e1a sharp65e1_front=$S/render5_${C8}_v65e1f bgvote65e2_all=$S/render5_${C8}_v65e2a bgvote65e2_front=$S/render5_${C8}_v65e2f"
env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $S/${C8}_v61s $S/pi_checks_${C8}_c.json 86 -5,40,-15,15 $VARS > $S/pi_checks_${C8}_c.log 2>&1
env OMP_NUM_THREADS=4 $PY a2_robust2.py $C8 $S/${C8}_v61s $VARS > $S/a2_robust2_${C8}_c.log 2>&1
NM=$(ls $SM/render5_${C8}_v65e[12][af]/worldmap.npz 2>/dev/null | wc -l)
echo "ZZV65EN-maps$NM-checks$(grep -c ZZPICHECKS $S/pi_checks_${C8}_c.log)-a2r$(grep -c ZZA2R $S/a2_robust2_${C8}_c.log)ZZ" >> $LOG
