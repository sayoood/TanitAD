#!/bin/bash
# Thor: pre-registered arm d1 (renderer v5j docstring) = box-free d0 + OCC_FILL_M=2.5, one clip, ALL 7 and FRONT; scores; checks.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
C8=$1; J=$2; LOG=$SM/v65d1.log
cd $SM/eval
for ARM in a f; do
  SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 OMP_NUM_THREADS=4 \
      OCC_FILL_M=2.5 $PY sam3map_render_v5j.py $C8 $SRC $SM/render5_${C8}_v65d1$ARM > $SM/render5_${C8}_v65d1$ARM.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65d1$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65d1 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SRC > $SM/score_v65d1${ARM}_$C8.log 2>&1
done
S=$SM
VARS="nobox65d0_all=$S/render5_${C8}_v65d0a fill65d1_all=$S/render5_${C8}_v65d1a fill65d1_front=$S/render5_${C8}_v65d1f"
env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $S/${C8}_v61s $S/pi_checks_${C8}_h.json $J -5,40,-15,15 $VARS > $S/pi_checks_${C8}_h.log 2>&1
env OMP_NUM_THREADS=4 $PY a2_robust2.py $C8 $S/${C8}_v61s $VARS > $S/a2_robust2_${C8}_h.log 2>&1
NM=$(ls $SM/render5_${C8}_v65d1[af]/worldmap.npz 2>/dev/null | wc -l)
echo "ZZD1-$C8-maps$NM-checks$(grep -c ZZPICHECKS $S/pi_checks_${C8}_h.log)-a2r$(grep -c ZZA2R $S/a2_robust2_${C8}_h.log)ZZ" >> $LOG
