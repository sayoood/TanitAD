#!/bin/bash
# Thor: pre-registered v6z (front-camera 2x zoom tile, sam3map_zoom_fw.py) -> refine v6 -> consensus -> render v5 (nearest)
# -> LiDAR scores (clip vote on v6z, GT world map on v61z) -> H.264. Night first (the target), day second (no-regression control).
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
VIEWS=CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT
PP=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map
for C8 in 73495082f98b 4fbd97b6a4b7; do
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_zoom_fw.py $C8 $SM/${C8}_v6raw $SM/${C8}_v6zraw > $SM/zoom_fw_$C8.log 2>&1
  cd $SM
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS SAM3MAP_TAG=v6z PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > refine_v6z_$C8.log 2>&1
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_consensus.py $C8 $SM/${C8}_v6z $SM/${C8}_v61z > $SM/consensus_v61z_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_render_v5.py $C8 $SM/${C8}_v61z $SM/render5_${C8}_v61z > $SM/render5_${C8}_v61z.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_v6z > $SM/score_v6z_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v61z/worldmap.npz SAM3MAP_SCORE_TAG=gt OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_v61z > $SM/score_v61z_gt_$C8.log 2>&1
  gst-launch-1.0 -e multifilesrc location="$SM/render5_${C8}_v61z/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/sam3map_v61z_${C8}.mp4 > $SM/v61z_${C8}.gst.log 2>&1
  NZ=$(ls $SM/${C8}_v6zraw/[0-9][0-9][0-9].npz 2>/dev/null | wc -l); NS=$(stat -c %s $SM/sam3map_score_${C8}_v6z.json 2>/dev/null || echo 0)
  NG=$(stat -c %s $SM/sam3map_score_${C8}_v61z_gt.json 2>/dev/null || echo 0); NM=$(stat -c %s $SM/sam3map_v61z_${C8}.mp4 2>/dev/null || echo 0)
  echo "ZZV6Z-$C8-z$NZ-s$NS-g$NG-m${NM}ZZ" >> $SM/v6z_chain.log
done
echo "ZZV6ZCHAIN-ENDZZ" >> $SM/v6z_chain.log
