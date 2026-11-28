#!/bin/bash
# Thor: v6s = crosswalk STRIPES from SAM3 prompts -> refine -> consensus without gap promotion -> majority map (renderer v5b)
# for ALL 7 cameras and FRONT only -> LiDAR scores -> front-vs-all video. Night first (the PI's frame), day second.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
PP=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map
VIEWS=CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT
LOG=$SM/v6s_chain.log
for C8 in 73495082f98b 4fbd97b6a4b7; do
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_xwalk_stripes.py $C8 $SM/${C8}_v6raw $SM/${C8}_v6sraw > $SM/xwalk_stripes_$C8.log 2>&1
  NS=$(ls $SM/${C8}_v6sraw/[0-9][0-9][0-9].npz 2>/dev/null | wc -l)
  [ "$NS" -eq 96 ] || { echo "ZZV6S-$C8-STRIPES-INCOMPLETE-$NS""ZZ" >> $LOG; continue; }
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY make_front_only.py $C8 $SM/${C8}_v6sraw $SM/${C8}_v6sfraw > $SM/front_only_v6s_$C8.log 2>&1
  cd $SM
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS SAM3MAP_TAG=v6s PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > refine_v6s_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=CAM_FW SAM3MAP_TAG=v6sf PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > refine_v6sf_$C8.log 2>&1
  cd $SM/eval
  for ARM in a f; do
    TG=v6s$([ $ARM = f ] && echo f); OUTN=v61s$([ $ARM = f ] && echo f)
    env SAM3MAP_ROOT=$SM/native7 CONSENSUS_NO_PROMOTE=1 OMP_NUM_THREADS=5 $PY sam3map_consensus.py $C8 $SM/${C8}_$TG $SM/${C8}_$OUTN > $SM/consensus_${OUTN}_$C8.log 2>&1
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 OMP_NUM_THREADS=5 $PY sam3map_render_v5b.py $C8 $SM/${C8}_$OUTN $SM/render5_${C8}_v65s$ARM > $SM/render5_${C8}_v65s$ARM.log 2>&1
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65s$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65s OMP_NUM_THREADS=5 $PY thor_run.py score $C8 $SM/${C8}_$OUTN > $SM/score_v65s${ARM}_$C8.log 2>&1
  done
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=5 $PY compare_front_vs_all.py $C8 $SM/${C8}_v61sf $SM/render5_${C8}_v65sf $SM/${C8}_v61s $SM/render5_${C8}_v65sa $SM/cmp65s_front_vs_all_$C8 $SM/sam3map_score_${C8}_v61sf_gt_v65s.json $SM/sam3map_score_${C8}_v61s_gt_v65s.json > $SM/compare65s_$C8.log 2>&1
  gst-launch-1.0 -e multifilesrc location="$SM/cmp65s_front_vs_all_$C8/f%04d.png" index=0 caps="image/png,framerate=(fraction)5/1" ! pngdec ! videoconvert ! video/x-raw,format=I420 ! x264enc pass=qual quantizer=22 speed-preset=medium key-int-max=10 ! video/x-h264,profile=high ! h264parse ! mp4mux ! filesink location=$SM/compare65s_front_vs_all_$C8.mp4 > $SM/compare65s_$C8.gst.log 2>&1
  NP=$(ls $SM/cmp65s_front_vs_all_$C8/f*.png 2>/dev/null | wc -l); NM=$(stat -c %s $SM/compare65s_front_vs_all_$C8.mp4 2>/dev/null || echo 0)
  echo "ZZV6S-$C8-stripes$NS-png$NP-mp4${NM}ZZ" >> $LOG
done
echo "ZZV6S-ENDZZ" >> $LOG
