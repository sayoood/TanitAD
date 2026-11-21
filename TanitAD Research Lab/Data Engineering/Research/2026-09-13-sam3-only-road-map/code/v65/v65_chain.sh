#!/bin/bash
# Thor: renderer composite majority_v65 + stripe-only crosswalks for FRONT-ONLY (v61f npz) and ALL-7 (v61 npz) arms, LiDAR
# scores of both world maps, and the front-vs-all side-by-side video with the new maps. Night first, day second.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
LOG=$SM/v65_chain.log
for C8 in 73495082f98b 4fbd97b6a4b7; do
  cd $SM/eval
  for ARM in f a; do
    SRC=$SM/${C8}_v61$([ $ARM = f ] && echo f)
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 CROSSWALK_STRIPES=1 OMP_NUM_THREADS=5 $PY sam3map_render_v5.py $C8 $SRC $SM/render5_${C8}_v65$ARM > $SM/render5_${C8}_v65$ARM.log 2>&1
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65 OMP_NUM_THREADS=5 $PY thor_run.py score $C8 $SRC > $SM/score_v65${ARM}_$C8.log 2>&1
  done
  env SAM3MAP_ROOT=$SM/native7 CROSSWALK_STRIPES=1 OMP_NUM_THREADS=5 $PY compare_front_vs_all.py $C8 $SM/${C8}_v61f $SM/render5_${C8}_v65f $SM/${C8}_v61 $SM/render5_${C8}_v65a $SM/cmp65_front_vs_all_$C8 $SM/sam3map_score_${C8}_v61f_gt_v65.json $SM/sam3map_score_${C8}_v61_gt_v65.json > $SM/compare65_$C8.log 2>&1
  gst-launch-1.0 -e multifilesrc location="$SM/cmp65_front_vs_all_$C8/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/compare65_front_vs_all_$C8.mp4 > $SM/compare65_$C8.gst.log 2>&1
  NP=$(ls $SM/cmp65_front_vs_all_$C8/f*.png 2>/dev/null | wc -l); NM=$(stat -c %s $SM/compare65_front_vs_all_$C8.mp4 2>/dev/null || echo 0)
  NF=$(stat -c %s $SM/sam3map_score_${C8}_v61f_gt_v65.json 2>/dev/null || echo 0); NA=$(stat -c %s $SM/sam3map_score_${C8}_v61_gt_v65.json 2>/dev/null || echo 0)
  echo "ZZV65-$C8-png$NP-mp4$NM-sf$NF-sa${NA}ZZ" >> $LOG
done
echo "ZZV65-ENDZZ" >> $LOG
