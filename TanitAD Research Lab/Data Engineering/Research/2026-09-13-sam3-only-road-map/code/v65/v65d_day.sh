#!/bin/bash
# Thor: day clip after the v6.5c chain -- v6.5d (paint rule p95 + crosswalk stripes trimmed) ALL 7 and FRONT, LiDAR scores,
# PI checks over every day variant, robust A2, reprojection video of v6.5d. Night was run by v65d_night.sh.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
LOG=$SM/v65d_day.log
C8=4fbd97b6a4b7
until grep -q "ZZV65C-$C8" $SM/v65c_chain.log 2>/dev/null; do sleep 30; done
cd $SM/eval
for ARM in a f; do
  SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 AGENT_DRIVABLE=1 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 OMP_NUM_THREADS=4 $PY sam3map_render_v5d.py $C8 $SRC $SM/render5_${C8}_v65d$ARM > $SM/render5_${C8}_v65d$ARM.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65d$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65d OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SRC > $SM/score_v65d${ARM}_$C8.log 2>&1
done
S=$SM
VARS="nearest_all=$S/render5_${C8}_v61 nearest_front=$S/render5_${C8}_v61f stripes65s_all=$S/render5_${C8}_v65sa stripes65s_front=$S/render5_${C8}_v65sf agents65c_all=$S/render5_${C8}_v65ca agents65c_front=$S/render5_${C8}_v65cf paint65d_all=$S/render5_${C8}_v65da paint65d_front=$S/render5_${C8}_v65df"
env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $S/${C8}_v61s $S/pi_checks_${C8}_b.json 41 -5,40,-15,15 $VARS > $S/pi_checks_${C8}_b.log 2>&1
env OMP_NUM_THREADS=4 $PY a2_robust.py $C8 $S/${C8}_v61s $VARS > $S/a2_robust_${C8}.log 2>&1
cd $SM && ./reproj_video.sh $C8 v65d v61sf v61s $S/pi_checks_${C8}_b.json paint65d_front paint65d_all
NA=$(ls $SM/render5_${C8}_v65da/worldmap.npz $SM/render5_${C8}_v65df/worldmap.npz 2>/dev/null | wc -l)
NS=$(ls $SM/sam3map_score_${C8}_v61s_gt_v65d.json $SM/sam3map_score_${C8}_v61sf_gt_v65d.json 2>/dev/null | wc -l)
NP=$(grep -c ZZPICHECKS $S/pi_checks_${C8}_b.log); NR=$(grep -c ZZA2R $S/a2_robust_${C8}.log)
echo "ZZV65DD-maps$NA-scores$NS-checks$NP-a2r${NR}ZZ" >> $LOG
