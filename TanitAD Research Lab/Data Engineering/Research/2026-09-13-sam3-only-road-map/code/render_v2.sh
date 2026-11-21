#!/bin/bash
# Thor: wait for a clip's v1 extraction, then v2 refine (v1 rasters + one small SAM3 bonnet pass) -> render -> H.264.
C8=$1
SM=/home/nvidia/sam3map
OUT=$SM/render_${C8}_v2
gone=0
while :; do
  grep -qs ZZSAM3MAP-DONEZZ "$SM/run_${C8}.log" && break
  if [ -f "$SM/run_${C8}.log" ] && [ "$(ps -eo args | grep -c "[s]am3map_extract.py ${C8}")" -eq 0 ]; then
    gone=$((gone + 1))
    if [ $gone -ge 2 ]; then echo "v1 extractor ${C8} gone without its done marker" >> $OUT.render.log; break; fi
  else
    gone=0
  fi
  sleep 30
done
cd $SM || exit 1
PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python sam3map_refine.py ${C8} > refine_${C8}.log 2>&1
echo "refine_exit=$?" >> refine_${C8}.log
cd $SM/eval || exit 1
OMP_NUM_THREADS=4 /home/nvidia/venvs/tanitad-edge/bin/python thor_run.py render ${C8} $SM/${C8}_v2 $OUT >> $OUT.render.log 2>&1
echo "render_exit=$?" >> $OUT.render.log
nf=$(ls $OUT/f*.png 2>/dev/null | wc -l)
gst-launch-1.0 -e multifilesrc location="$OUT/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=21 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$OUT.mp4 > $OUT.gst.log 2>&1
echo "ZZV2-${nf}-$(stat -c %s $OUT.mp4 2>/dev/null || echo 0)ZZ" >> $OUT.render.log
