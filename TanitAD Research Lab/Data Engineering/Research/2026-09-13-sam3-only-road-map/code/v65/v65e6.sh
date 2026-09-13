#!/bin/bash
# Thor: pre-registered arm e6 (renderer v5h docstring) on BOTH clips, ALL 7 and FRONT: e5 + BG_VETO=road.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
C8=$1; J=$2; LOG=$SM/v65e6.log
cd $SM/eval
for ARM in a f; do
  SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 AGENT_DRIVABLE=1 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 OMP_NUM_THREADS=4 \
      BG_VOTE_NEAR=10 BG_EDGES=1 BG_MINCOMP=50 SPECK_FILL=50 BG_EDGE_OPEN=1 BG_VETO=road $PY sam3map_render_v5h.py $C8 $SRC $SM/render5_${C8}_v65e6$ARM > $SM/render5_${C8}_v65e6$ARM.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65e6$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65e6 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SRC > $SM/score_v65e6${ARM}_$C8.log 2>&1
done
S=$SM
VARS="paint65d_all=$S/render5_${C8}_v65da noopen65e5_all=$S/render5_${C8}_v65e5a veto65e6_all=$S/render5_${C8}_v65e6a veto65e6_front=$S/render5_${C8}_v65e6f"
env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $S/${C8}_v61s $S/pi_checks_${C8}_f.json $J -5,40,-15,15 $VARS > $S/pi_checks_${C8}_f.log 2>&1
env OMP_NUM_THREADS=4 $PY a2_robust2.py $C8 $S/${C8}_v61s $VARS > $S/a2_robust2_${C8}_f.log 2>&1
NM=$(ls $SM/render5_${C8}_v65e6[af]/worldmap.npz 2>/dev/null | wc -l)
echo "ZZV65E6-$C8-maps$NM-checks$(grep -c ZZPICHECKS $S/pi_checks_${C8}_f.log)-a2r$(grep -c ZZA2R $S/a2_robust2_${C8}_f.log)ZZ" >> $LOG
