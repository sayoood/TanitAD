#!/bin/bash
# Thor: fields with NEAR_PAINT_M=12 (renderer v5n) for one clip, ALL 7 and FRONT; then composition with the validated options
# (LINE_EDGE_GAP=1 LINE_MIN_LEN_M=1.0 WLK_ISLAND_M2=3 EDGE_OBS=near XWALK=dirclose XWALK_LEN=11) as v65u; same composition on the d0 fields as v65r.
SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python
C8=$1; LOG=$SM/v65t_near.log
cd $SM/eval
for ARM in a f; do
  SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 FIELDS=1 NEAR_PAINT_M=12 OMP_NUM_THREADS=4 \
      $PY sam3map_render_v5n.py $C8 $SRC $SM/render5_${C8}_v65t$ARM > $SM/render5_${C8}_v65t$ARM.log 2>&1
  env LINE_EDGE_GAP=1 LINE_MIN_LEN_M=1.0 WLK_ISLAND_M2=3 EDGE_OBS=near XWALK=dirclose XWALK_LEN=11 $PY compose.py $SM/render5_${C8}_v65t$ARM $SM/render5_${C8}_v65u$ARM > $SM/compose_v65u${ARM}_$C8.log 2>&1
  env LINE_EDGE_GAP=1 LINE_MIN_LEN_M=1.0 WLK_ISLAND_M2=3 EDGE_OBS=near XWALK=dirclose XWALK_LEN=11 $PY compose.py $SM/render5_${C8}_v65m$ARM $SM/render5_${C8}_v65r$ARM > $SM/compose_v65r${ARM}_$C8.log 2>&1
done
env OMP_NUM_THREADS=4 $PY xwalk_reproj_contrast.py $C8 $SM/${C8}_v61s $SM/render5_${C8}_v65ma "d0 delivered=$SM/render5_${C8}_v65d0a" "r fixes+dirclose (d0 fields)=$SM/render5_${C8}_v65ra" "u fixes+dirclose (near-paint fields)=$SM/render5_${C8}_v65ua" > $SM/xrc_tu_$C8.log 2>&1
echo "ZZNEAR-$C8-$(ls $SM/render5_${C8}_v65u[af]/worldmap.npz $SM/render5_${C8}_v65r[af]/worldmap.npz 2>/dev/null | wc -l)-$(grep -c ZZXRC $SM/xrc_tu_$C8.log)ZZ" >> $LOG
