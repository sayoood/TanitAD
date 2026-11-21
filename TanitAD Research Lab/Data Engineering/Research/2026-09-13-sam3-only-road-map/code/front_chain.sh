#!/bin/bash
# Thor: after the two-clip v3 extraction frees the GPU, run the front-only v3 pipeline on the mined clips and join them.
SM=/home/nvidia/sam3map
CLIPS="$@"
while ! grep -qs ZZV3CHAIN-DONEZZ $SM/v3_extract_chain.log; do
  [ "$(ps -eo args | grep -c "[v]3_extract_chain.sh")" -eq 0 ] && [ "$(ps -eo args | grep -c "[s]am3map_extract_v3.py")" -eq 0 ] && break
  sleep 60
done
while [ "$(ps -eo args | grep -c "[b]uild_front_seq.py")" -gt 0 ]; do sleep 20; done
ENVF="SAM3MAP_ROOT=$SM/front SAM3MAP_VIEWS=CAM_F0"
for C8 in $CLIPS; do
  cd $SM
  env $ENVF PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 /home/nvidia/venvs/tanitad-edge/bin/python sam3map_extract_v3.py $C8 all > run_v3_$C8.log 2>&1
  env $ENVF PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python sam3map_refine_v3.py $C8 > refine_v3_$C8.log 2>&1
  cd $SM/eval && env SAM3MAP_ROOT=$SM/front OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py render3 $C8 $SM/${C8}_v3 $SM/render_${C8}_v3 > $SM/render_${C8}_v3.render.log 2>&1
  echo "clip $C8 done: $(ls $SM/render_${C8}_v3/f*.png 2>/dev/null | wc -l) frames" >> $SM/front_chain.log
done
cd $SM/eval && /home/nvidia/venvs/tanitad-edge/bin/python make_long.py $SM/long_v3 $CLIPS >> $SM/front_chain.log 2>&1
gst-launch-1.0 -e multifilesrc location="$SM/long_v3/f%05d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/long_v3.mp4 > $SM/long_v3.gst.log 2>&1
echo "ZZLONG-$(ls $SM/long_v3/f*.png 2>/dev/null | wc -l)-$(stat -c %s $SM/long_v3.mp4 2>/dev/null || echo 0)ZZ" >> $SM/front_chain.log
