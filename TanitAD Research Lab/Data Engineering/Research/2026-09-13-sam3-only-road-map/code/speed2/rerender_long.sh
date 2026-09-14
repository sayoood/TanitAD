#!/bin/bash
# re-render the long video with the fixed panel titles (same maps, same bars) into long_frames2 -> sam3map_fast_long_v2.mp4
SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python; cd $SM/eval
FL="fp16 encoder + batched prompts + async + streaming"; FT=spdS4; FT0=spdF4a
FRN="1f1f05ca011d b5d9b91e6637 b975bf8ebf95 41f10d46174e 26015e788849 f63e215a546a c1dc66b6ae42 d672fc17a315"
ORDER="73495082f98b:native7:CAM_FW:$FT0:spd0:LiDAR 4fbd97b6a4b7:native7:CAM_FW:$FT0:spd0:LiDAR 6924358fafe0:native7:CAM_FW:$FT:spdSA:LiDAR 0d90d20036a3:native7:CAM_FW:$FT:spdSA:LiDAR"
for C8 in $FRN; do ORDER="$ORDER $C8:front_native:CAM_F0:$FT:spdSA:ego-path"; done
N=$(echo $ORDER | wc -w); i=0; run=0
for item in $ORDER; do
  i=$((i+1)); IFS=: read C8 R CAM FTAG RTAG GR <<< "$item"
  GL=$([ "$GR" = "LiDAR" ] && echo "LiDAR" || echo "ego path, no LiDAR")
  env OMP_NUM_THREADS=2 $PY render_fast_long.py $SM/long_frames2/$(printf %02d $i)_$C8 $SM/$R $CAM $C8 $FTAG $RTAG "$FL" $SM/long_bars.json $i $N "$GL" > $SM/long_render2_$C8.log 2>&1 &
  run=$((run+1)); if [ $run -ge 4 ]; then wait -n; run=$((run-1)); fi
done
wait
LONG=$SM/long_fast2; mkdir -p $LONG; k=0
for d in $(ls -d $SM/long_frames2/*/ | sort); do for f in $(ls $d/f*.jpg | sort); do ln -sf $f $LONG/f$(printf %05d $k).jpg; k=$((k+1)); done; done
gst-launch-1.0 -e multifilesrc location="$LONG/f%05d.jpg" index=0 caps="image/jpeg,framerate=(fraction)5/1" ! jpegdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/sam3map_fast_long_v2.mp4 > $SM/sam3map_fast_long_v2.gst.log 2>&1
echo "ZZRERENDER-$(grep -l ZZRENDERLONG $SM/long_render2_*.log | wc -l)-frames$k-mp4$(stat -c %s $SM/sam3map_fast_long_v2.mp4 2>/dev/null || echo 0)ZZ" > $SM/rerender_long.done
