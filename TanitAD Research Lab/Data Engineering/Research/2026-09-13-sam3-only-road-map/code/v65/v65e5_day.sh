#!/bin/bash
# Thor: arm e5 (renderer v5g) on the DAY clip, ALL 7 and FRONT, after it passed its pre-registered night bars.
# DAY BARS, PRE-REGISTERED 2026-09-13 before this run (no-regression form; day v6.5d already puts < 1 % of robust obstacle
# points on drivable): e5 WORKS on the day clip (all 7) if A2r_onroad <= day v6.5d's A2r_onroad (same run) AND fragments
# <= 105.1 per 1000 m^2 AND LiDAR curb recall >= 0.667 AND crosswalk coloured share <= 0.60 (day v6.5d: 105.1 / 0.687 / 0.564).
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
LOG=$SM/v65e5_day.log
C8=4fbd97b6a4b7
cd $SM/eval
for ARM in a f; do
  SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 AGENT_DRIVABLE=1 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 OMP_NUM_THREADS=4 \
      BG_VOTE_NEAR=10 BG_EDGES=1 BG_MINCOMP=50 SPECK_FILL=50 BG_EDGE_OPEN=1 $PY sam3map_render_v5g.py $C8 $SRC $SM/render5_${C8}_v65e5$ARM > $SM/render5_${C8}_v65e5$ARM.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65e5$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65e5 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SRC > $SM/score_v65e5${ARM}_$C8.log 2>&1
done
S=$SM
VARS="nearest_all=$S/render5_${C8}_v61 paint65d_all=$S/render5_${C8}_v65da paint65d_front=$S/render5_${C8}_v65df noopen65e5_all=$S/render5_${C8}_v65e5a noopen65e5_front=$S/render5_${C8}_v65e5f"
env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $S/${C8}_v61s $S/pi_checks_${C8}_e.json 41 -5,40,-15,15 $VARS > $S/pi_checks_${C8}_e.log 2>&1
env OMP_NUM_THREADS=4 $PY a2_robust2.py $C8 $S/${C8}_v61s $VARS > $S/a2_robust2_${C8}_e.log 2>&1
NM=$(ls $SM/render5_${C8}_v65e5[af]/worldmap.npz 2>/dev/null | wc -l)
echo "ZZV65E5D-maps$NM-checks$(grep -c ZZPICHECKS $S/pi_checks_${C8}_e.log)-a2r$(grep -c ZZA2R $S/a2_robust2_${C8}_e.log)ZZ" >> $LOG
