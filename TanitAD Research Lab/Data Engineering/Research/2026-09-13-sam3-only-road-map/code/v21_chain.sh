#!/bin/bash
# Thor: v2.1 (component-fraction patch rule) for both clips: set the v2.0 outputs aside, refine, render, encode, score.
SM=/home/nvidia/sam3map
ENVP="PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4"
for C8 in 4fbd97b6a4b7 73495082f98b; do
  cd $SM || exit 1
  for p in ${C8}_v2 render_${C8}_v2 render_${C8}_v2.mp4 render_${C8}_v2.render.log refine_${C8}.log refine_${C8}.json; do
    [ -e "$p" ] && mv "$p" "v20_$p"
  done
  env $ENVP /home/nvidia/venvs/tanitad-edge/bin/python sam3map_refine.py ${C8} > refine_${C8}.log 2>&1
  echo "refine_exit=$?" >> refine_${C8}.log
  OUT=$SM/render_${C8}_v2
  cd $SM/eval || exit 1
  OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py render ${C8} $SM/${C8}_v2 $OUT > $OUT.render.log 2>&1
  echo "render_exit=$?" >> $OUT.render.log
  gst-launch-1.0 -e multifilesrc location="$OUT/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=21 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$OUT.mp4 > $OUT.gst.log 2>&1
  OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py score ${C8} $SM/${C8}_v2 > $SM/score_${C8}_v2.log 2>&1
  echo "score_exit=$?" >> $SM/score_${C8}_v2.log
done
echo ZZV21-DONEZZ >> $SM/v21_chain.log
