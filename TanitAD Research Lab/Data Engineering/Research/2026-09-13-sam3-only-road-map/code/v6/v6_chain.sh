#!/bin/bash
# Thor: SAM3 map v6 on ALL 7 NATIVE cameras for the two LiDAR clips: extract -> refine (per-camera ego masks) -> v4 BEV
# render -> H.264 -> LiDAR score (camera-model coverage). One ZZ line per clip in v6_chain.log.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
VIEWS=CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT
for C8 in "$@"; do
  cd $SM
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_extract_v6.py $C8 all > run_v6_$C8.log 2>&1
  NX=$(ls $SM/${C8}_v6raw/[0-9][0-9][0-9].npz 2>/dev/null | wc -l)
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS SAM3MAP_TAG=v6 PYTHONPATH=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > refine_v6_$C8.log 2>&1
  NR=$(ls $SM/${C8}_v6/[0-9][0-9][0-9].npz 2>/dev/null | wc -l)
  cd $SM/eval && env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY thor_run.py render4 $C8 $SM/${C8}_v6 $SM/render_${C8}_v6 > $SM/render_${C8}_v6.render.log 2>&1
  NP=$(ls $SM/render_${C8}_v6/f*.png 2>/dev/null | wc -l)
  gst-launch-1.0 -e multifilesrc location="$SM/render_${C8}_v6/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/sam3map_v6_${C8}.mp4 > $SM/v6_${C8}.gst.log 2>&1
  NM=$(stat -c %s $SM/sam3map_v6_${C8}.mp4 2>/dev/null || echo 0)
  cd $SM/eval && env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_v6 > $SM/score_v6_$C8.log 2>&1
  NS=$(stat -c %s $SM/sam3map_score_${C8}_v6.json 2>/dev/null || echo 0)
  echo "ZZV6-$C8-x$NX-r$NR-p$NP-m$NM-s${NS}ZZ" >> $SM/v6_chain.log
done
echo "ZZV6CHAIN-ENDZZ" >> $SM/v6_chain.log
