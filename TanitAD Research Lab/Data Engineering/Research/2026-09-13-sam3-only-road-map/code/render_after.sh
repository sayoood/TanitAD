#!/bin/bash
# Thor: wait for one clip's SAM3 extraction to end, render every frame on the CPU, encode H.264 with GStreamer x264enc.
C8=$1
SM=/home/nvidia/sam3map
LOG=$SM/run_${C8}.log
OUT=$SM/render_${C8}
gone=0
while :; do
  grep -qs ZZSAM3MAP-DONEZZ "$LOG" && break
  if [ -f "$LOG" ] && [ "$(ps -eo args | grep -c "[s]am3map_extract.py ${C8}")" -eq 0 ]; then
    gone=$((gone + 1))
    if [ $gone -ge 2 ]; then echo "extractor ${C8} gone without its done marker" >> $OUT.render.log; break; fi
  else
    gone=0
  fi
  sleep 30
done
cd $SM/eval || exit 1
OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py render ${C8} $SM/${C8} $OUT >> $OUT.render.log 2>&1
echo "render_exit=$?" >> $OUT.render.log
nf=$(ls $OUT/f*.png 2>/dev/null | wc -l)
gst-launch-1.0 -e multifilesrc location="$OUT/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=21 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$OUT.mp4 > $OUT.gst.log 2>&1
echo "ZZRENDER-${nf}-$(stat -c %s $OUT.mp4 2>/dev/null || echo 0)ZZ" >> $OUT.render.log
