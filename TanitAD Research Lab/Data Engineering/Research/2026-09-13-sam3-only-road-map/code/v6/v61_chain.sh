#!/bin/bash
# Thor: v6.1 = v6 refined maps + R4 clip consensus -> v5 metric BEV render -> H.264 -> BEV-head GT export -> LiDAR score.
# Waits for each clip's refine summary (written last by sam3map_refine_v6.py) before starting that clip.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
mkdir -p $SM/gt_v61
for C8 in "$@"; do
  until [ -s $SM/refine_v6_$C8.json ]; do sleep 30; done
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_consensus.py $C8 $SM/${C8}_v6 $SM/${C8}_v61 > $SM/consensus_v61_$C8.log 2>&1
  NC=$(ls $SM/${C8}_v61/[0-9][0-9][0-9].npz 2>/dev/null | wc -l)
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_render_v5.py $C8 $SM/${C8}_v61 $SM/render5_${C8}_v61 > $SM/render5_${C8}_v61.log 2>&1
  NP=$(ls $SM/render5_${C8}_v61/f*.png 2>/dev/null | wc -l)
  gst-launch-1.0 -e multifilesrc location="$SM/render5_${C8}_v61/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/sam3map_v61_${C8}.mp4 > $SM/v61_${C8}.gst.log 2>&1
  NM=$(stat -c %s $SM/sam3map_v61_${C8}.mp4 2>/dev/null || echo 0)
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY sam3map_export_gt.py $C8 $SM/${C8}_v61 $SM/render5_${C8}_v61 $SM/gt_v61 > $SM/export_v61_$C8.log 2>&1
  NG=$(grep -c ZZEXPORTGT-OK $SM/export_v61_$C8.log)
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_v61 > $SM/score_v61_$C8.log 2>&1
  NS=$(stat -c %s $SM/sam3map_score_${C8}_v61.json 2>/dev/null || echo 0)
  echo "ZZV61-$C8-c$NC-p$NP-m$NM-g$NG-s${NS}ZZ" >> $SM/v61_chain.log
done
echo "ZZV61CHAIN-ENDZZ" >> $SM/v61_chain.log
