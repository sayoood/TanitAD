#!/bin/bash
# Thor: after the front-only long-video chain, run v3.1 on both 7-camera clips: extract -> refine -> render -> H.264 -> score.
SM=/home/nvidia/sam3map
while ! grep -qs "^ZZLONG" $SM/front_chain.log; do
  [ "$(ps -eo args | grep -c "[f]ront_chain.sh")" -eq 0 ] && break
  sleep 60
done
for C8 in 73495082f98b 4fbd97b6a4b7; do
  cd $SM
  PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 /home/nvidia/venvs/tanitad-edge/bin/python sam3map_extract_v31.py $C8 all > run_v31_$C8.log 2>&1
  SAM3MAP_TAG=v31 PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python sam3map_refine_v3.py $C8 > refine_v31_$C8.log 2>&1
  OUT=$SM/render_${C8}_v31
  cd $SM/eval && OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py render3 $C8 $SM/${C8}_v31 $OUT > $OUT.render.log 2>&1
  gst-launch-1.0 -e multifilesrc location="$OUT/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=21 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$OUT.mp4 > $OUT.gst.log 2>&1
  echo "ZZV31VIDEO-$(ls $OUT/f*.png 2>/dev/null | wc -l)-$(stat -c %s $OUT.mp4 2>/dev/null || echo 0)ZZ" >> $OUT.render.log
  OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py score $C8 $SM/${C8}_v31 > $SM/score_${C8}_v31.log 2>&1
done
echo ZZV31CHAIN-DONEZZ >> $SM/v31_chain.log
