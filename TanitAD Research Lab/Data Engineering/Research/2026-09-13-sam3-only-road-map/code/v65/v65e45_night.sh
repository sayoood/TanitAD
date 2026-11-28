#!/bin/bash
# Thor: pre-registered arms e4 (e3 + SPECK_FILL=50) and e5 (e4 + BG_EDGE_OPEN=1), renderer v5g docstring, night clip, ALL 7 and FRONT.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
LOG=$SM/v65e45_night.log
C8=73495082f98b
cd $SM/eval
for E in e4 e5; do
  EXTRA=$([ $E = e5 ] && echo "BG_EDGE_OPEN=1" || echo "BG_EDGE_OPEN=3")
  for ARM in a f; do
    SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 AGENT_DRIVABLE=1 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 OMP_NUM_THREADS=4 \
        BG_VOTE_NEAR=10 BG_EDGES=1 BG_MINCOMP=50 SPECK_FILL=50 $EXTRA $PY sam3map_render_v5g.py $C8 $SRC $SM/render5_${C8}_v65${E}$ARM > $SM/render5_${C8}_v65${E}$ARM.log 2>&1
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65${E}$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65$E OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SRC > $SM/score_v65${E}${ARM}_$C8.log 2>&1
  done
done
S=$SM
VARS="paint65d_all=$S/render5_${C8}_v65da bgedge65e3_all=$S/render5_${C8}_v65e3a speck65e4_all=$S/render5_${C8}_v65e4a speck65e4_front=$S/render5_${C8}_v65e4f noopen65e5_all=$S/render5_${C8}_v65e5a noopen65e5_front=$S/render5_${C8}_v65e5f"
env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $S/${C8}_v61s $S/pi_checks_${C8}_e.json 86 -5,40,-15,15 $VARS > $S/pi_checks_${C8}_e.log 2>&1
env OMP_NUM_THREADS=4 $PY a2_robust2.py $C8 $S/${C8}_v61s $VARS > $S/a2_robust2_${C8}_e.log 2>&1
NM=$(ls $SM/render5_${C8}_v65e[45][af]/worldmap.npz 2>/dev/null | wc -l)
echo "ZZV65E45N-maps$NM-checks$(grep -c ZZPICHECKS $S/pi_checks_${C8}_e.log)-a2r$(grep -c ZZA2R $S/a2_robust2_${C8}_e.log)ZZ" >> $LOG
