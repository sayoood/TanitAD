#!/bin/bash
# Thor: SAM3 map v5 on the NATIVE 120 deg front camera for the mined clips: extract -> refine -> v4 BEV render -> long video.
SM=/home/nvidia/sam3map
CLIPS="$@"
ENVF="SAM3MAP_ROOT=$SM/front_native SAM3MAP_VIEWS=CAM_F0"
PYP="PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1"
for C8 in $CLIPS; do
  cd $SM
  env $ENVF $PYP OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python sam3map_extract_v5.py $C8 all > run_v5_$C8.log 2>&1
  env $ENVF $PYP SAM3MAP_TAG=v5 OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python sam3map_refine_v5.py $C8 > refine_v5_$C8.log 2>&1
  cd $SM/eval && env SAM3MAP_ROOT=$SM/front_native OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py render4 $C8 $SM/${C8}_v5 $SM/render_${C8}_v5 > $SM/render_${C8}_v5.render.log 2>&1
  echo "clip $C8 v5: $(ls $SM/render_${C8}_v5/f*.png 2>/dev/null | wc -l) frames; $(grep -a -o "totals.*" $SM/refine_v5_$C8.log | cut -c1-120)" >> $SM/v5_front_chain.log
done
cd $SM/eval && env SAM3MAP_LONG_TAG=v5 SAM3MAP_ROOT=$SM/front_native /home/nvidia/venvs/tanitad-edge/bin/python make_long.py $SM/long_v5 $CLIPS >> $SM/v5_front_chain.log 2>&1
gst-launch-1.0 -e multifilesrc location="$SM/long_v5/f%05d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/long_v5.mp4 > $SM/long_v5.gst.log 2>&1
echo "ZZLONGV5-$(ls $SM/long_v5/f*.png 2>/dev/null | wc -l)-$(stat -c %s $SM/long_v5.mp4 2>/dev/null || echo 0)ZZ" >> $SM/v5_front_chain.log
