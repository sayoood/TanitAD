#!/bin/bash
# Thor: for each clip, once its v3 extraction is done: refine_v3 -> render_v3 -> H.264 -> LiDAR score.
SM=/home/nvidia/sam3map
for C8 in 73495082f98b 4fbd97b6a4b7; do
  gone=0
  while ! grep -qs ZZSAM3MAP-DONEZZ $SM/run_v3_$C8.log; do
    if [ -f $SM/run_v3_$C8.log ] && [ "$(ps -eo args | grep -c "[s]am3map_extract_v3.py $C8")" -eq 0 ]; then
      gone=$((gone + 1)); [ $gone -ge 2 ] && { echo "v3 extractor $C8 gone without marker" >> $SM/v3_post_$C8.log; break; }
    else gone=0; fi
    sleep 30
  done
  cd $SM && PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python sam3map_refine_v3.py $C8 > refine_v3_$C8.log 2>&1
  echo "refine_exit=$?" >> refine_v3_$C8.log
  OUT=$SM/render_${C8}_v3
  cd $SM/eval && OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py render3 $C8 $SM/${C8}_v3 $OUT > $OUT.render.log 2>&1
  echo "render_exit=$?" >> $OUT.render.log
  gst-launch-1.0 -e multifilesrc location="$OUT/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=21 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$OUT.mp4 > $OUT.gst.log 2>&1
  echo "ZZV3VIDEO-$(ls $OUT/f*.png 2>/dev/null | wc -l)-$(stat -c %s $OUT.mp4 2>/dev/null || echo 0)ZZ" >> $OUT.render.log
  OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py score $C8 $SM/${C8}_v3 > $SM/score_${C8}_v3.log 2>&1
  echo "score_exit=$?" >> $SM/score_${C8}_v3.log
done
echo ZZV3POST-DONEZZ >> $SM/v3_post_chain.log
