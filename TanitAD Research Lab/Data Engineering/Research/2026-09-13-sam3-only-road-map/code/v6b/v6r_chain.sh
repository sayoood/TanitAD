#!/bin/bash
# Thor: pre-registered v6r (native CAM_FW drivable decision from the rectified virtual view, sam3map_rectified_road.py)
# -> refine v6 -> clip-vote LiDAR score; then consensus -> render v5 -> GT world-map score. Night first, day second.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
VIEWS=CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT
PP=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map
for C8 in 73495082f98b 4fbd97b6a4b7; do
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS OMP_NUM_THREADS=6 $PY sam3map_rectified_road.py $C8 $SM/${C8}_v6raw $SM/${C8}_v3raw $SM/${C8}_v6rraw > $SM/rectified_road_$C8.log 2>&1
  cd $SM
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS SAM3MAP_TAG=v6r PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > refine_v6r_$C8.log 2>&1
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_v6r > $SM/score_v6r_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_consensus.py $C8 $SM/${C8}_v6r $SM/${C8}_v61r > $SM/consensus_v61r_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_render_v5.py $C8 $SM/${C8}_v61r $SM/render5_${C8}_v61r > $SM/render5_${C8}_v61r.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v61r/worldmap.npz SAM3MAP_SCORE_TAG=gt OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_v61r > $SM/score_v61r_gt_$C8.log 2>&1
  NR=$(ls $SM/${C8}_v6rraw/[0-9][0-9][0-9].npz 2>/dev/null | wc -l); NS=$(stat -c %s $SM/sam3map_score_${C8}_v6r.json 2>/dev/null || echo 0)
  NG=$(stat -c %s $SM/sam3map_score_${C8}_v61r_gt.json 2>/dev/null || echo 0)
  echo "ZZV6R-$C8-r$NR-s$NS-g${NG}ZZ" >> $SM/v6r_chain.log
done
echo "ZZV6RCHAIN-ENDZZ" >> $SM/v6r_chain.log
