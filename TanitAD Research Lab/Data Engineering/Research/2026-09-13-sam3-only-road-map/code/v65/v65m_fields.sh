#!/bin/bash
# Thor: d0 re-rendered with FIELDS=1 (renderer v5m) on one clip, ALL 7 and FRONT; the world map must equal d0 cell for cell.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
C8=$1; LOG=$SM/v65m_fields.log
cd $SM/eval
for ARM in a f; do
  SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 FIELDS=1 OMP_NUM_THREADS=4 \
      $PY sam3map_render_v5m.py $C8 $SRC $SM/render5_${C8}_v65m$ARM > $SM/render5_${C8}_v65m$ARM.log 2>&1
  EQ=$($PY -c "import numpy as np; a=np.load('$SM/render5_${C8}_v65m$ARM/worldmap.npz')['cls']; b=np.load('$SM/render5_${C8}_v65d0$ARM/worldmap.npz')['cls']; print('EQUAL' if a.shape==b.shape and (a==b).all() else 'DIFF%d' % int((a!=b).sum()))")
  echo "ZZFIELDS-$C8-$ARM-$EQ-$(ls $SM/render5_${C8}_v65m$ARM/worldmap_fields.npz 2>/dev/null | wc -l)ZZ" >> $LOG
done
