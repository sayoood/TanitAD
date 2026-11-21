#!/bin/bash
# Thor: v6.5c = v6s data (stripe crosswalks) rendered with renderer v5c (majority + AGENT_DRIVABLE + SURFACE_MODE 5 +
# PAINT_SHARE 0.2) for ALL 7 and FRONT only, LiDAR scores, front-vs-all video. Starts each clip after the v6s chain has
# finished that clip (its ZZ line), so the two chains never share inputs mid-write.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
LOG=$SM/v65c_chain.log
for C8 in 73495082f98b 4fbd97b6a4b7; do
  until grep -q "ZZV6S-$C8" $SM/v6s_chain.log 2>/dev/null; do sleep 30; done
  cd $SM/eval
  for ARM in a f; do
    SRC=$SM/${C8}_v61s$([ $ARM = f ] && echo f)
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 AGENT_DRIVABLE=1 SURFACE_MODE=5 PAINT_SHARE=0.2 OMP_NUM_THREADS=5 $PY sam3map_render_v5c.py $C8 $SRC $SM/render5_${C8}_v65c$ARM > $SM/render5_${C8}_v65c$ARM.log 2>&1
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65c$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65c OMP_NUM_THREADS=5 $PY thor_run.py score $C8 $SRC > $SM/score_v65c${ARM}_$C8.log 2>&1
  done
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=5 $PY compare_front_vs_all.py $C8 $SM/${C8}_v61sf $SM/render5_${C8}_v65cf $SM/${C8}_v61s $SM/render5_${C8}_v65ca $SM/cmp65c_front_vs_all_$C8 $SM/sam3map_score_${C8}_v61sf_gt_v65c.json $SM/sam3map_score_${C8}_v61s_gt_v65c.json > $SM/compare65c_$C8.log 2>&1
  gst-launch-1.0 -e multifilesrc location="$SM/cmp65c_front_vs_all_$C8/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/compare65c_front_vs_all_$C8.mp4 > $SM/compare65c_$C8.gst.log 2>&1
  NP=$(ls $SM/cmp65c_front_vs_all_$C8/f*.png 2>/dev/null | wc -l); NM=$(stat -c %s $SM/compare65c_front_vs_all_$C8.mp4 2>/dev/null || echo 0)
  echo "ZZV65C-$C8-png$NP-mp4${NM}ZZ" >> $LOG
done
echo "ZZV65C-ENDZZ" >> $LOG
