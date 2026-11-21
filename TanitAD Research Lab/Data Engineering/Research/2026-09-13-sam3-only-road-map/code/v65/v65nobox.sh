#!/bin/bash
# Thor: BOX-FREE maps (PI: "you don't need the boxes") on one clip, ALL 7 and FRONT:
#   d0   = v6.5d without AGENT_DRIVABLE (renderer v5d)                              -- the PI-directed configuration, measured
#   e6nb = e6 with BG_OCC_BOXES=0 and without AGENT_DRIVABLE (renderer v5i docstring) -- pre-registered, both clips required
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
C8=$1; J=$2; LOG=$SM/v65nobox.log
cd $SM/eval
COMMON="SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 OMP_NUM_THREADS=4"
for ARM in a f; do
  SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
  env $COMMON $PY sam3map_render_v5d.py $C8 $SRC $SM/render5_${C8}_v65d0$ARM > $SM/render5_${C8}_v65d0$ARM.log 2>&1
  env $COMMON BG_VOTE_NEAR=10 BG_EDGES=1 BG_MINCOMP=50 SPECK_FILL=50 BG_EDGE_OPEN=1 BG_VETO=road BG_OCC_BOXES=0 $PY sam3map_render_v5i.py $C8 $SRC $SM/render5_${C8}_v65e6nb$ARM > $SM/render5_${C8}_v65e6nb$ARM.log 2>&1
  for V in d0 e6nb; do
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65$V$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65$V OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SRC > $SM/score_v65$V${ARM}_$C8.log 2>&1
  done
done
S=$SM
VARS="nearest_all=$S/render5_${C8}_v61 paint65d_all=$S/render5_${C8}_v65da nobox65d0_all=$S/render5_${C8}_v65d0a nobox65d0_front=$S/render5_${C8}_v65d0f nobox65e6nb_all=$S/render5_${C8}_v65e6nba nobox65e6nb_front=$S/render5_${C8}_v65e6nbf"
env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $S/${C8}_v61s $S/pi_checks_${C8}_g.json $J -5,40,-15,15 $VARS > $S/pi_checks_${C8}_g.log 2>&1
env OMP_NUM_THREADS=4 $PY a2_robust2.py $C8 $S/${C8}_v61s $VARS > $S/a2_robust2_${C8}_g.log 2>&1
NM=$(ls $SM/render5_${C8}_v65d0[af]/worldmap.npz $SM/render5_${C8}_v65e6nb[af]/worldmap.npz 2>/dev/null | wc -l)
echo "ZZNOBOX-$C8-maps$NM-checks$(grep -c ZZPICHECKS $S/pi_checks_${C8}_g.log)-a2r$(grep -c ZZA2R $S/a2_robust2_${C8}_g.log)ZZ" >> $LOG
