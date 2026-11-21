#!/bin/bash
# Thor: pre-registered v6n (rectified drivable decision for FW/CL/CR/RL/RR + tele drivable dropped) -> refine v6 -> clip-vote
# score; consensus -> render v5 -> GT world-map score. Waits for the v6r chain to finish. Night first, day second.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
VIEWS=CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT
PP=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map
until grep -q "ZZV6RCHAIN-END" $SM/v6r_chain.log 2>/dev/null; do sleep 30; done
for C8 in 73495082f98b 4fbd97b6a4b7; do
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS RECT_ALL=1 DROP_TELE=1 OMP_NUM_THREADS=6 $PY sam3map_rectified_road.py $C8 $SM/${C8}_v6raw $SM/${C8}_v3raw $SM/${C8}_v6nraw > $SM/rectified_all_$C8.log 2>&1
  cd $SM
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS SAM3MAP_TAG=v6n PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > refine_v6n_$C8.log 2>&1
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_v6n > $SM/score_v6n_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_consensus.py $C8 $SM/${C8}_v6n $SM/${C8}_v61n > $SM/consensus_v61n_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_render_v5.py $C8 $SM/${C8}_v61n $SM/render5_${C8}_v61n > $SM/render5_${C8}_v61n.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v61n/worldmap.npz SAM3MAP_SCORE_TAG=gt OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_v61n > $SM/score_v61n_gt_$C8.log 2>&1
  NR=$(ls $SM/${C8}_v6nraw/[0-9][0-9][0-9].npz 2>/dev/null | wc -l); NS=$(stat -c %s $SM/sam3map_score_${C8}_v6n.json 2>/dev/null || echo 0)
  NG=$(stat -c %s $SM/sam3map_score_${C8}_v61n_gt.json 2>/dev/null || echo 0)
  echo "ZZV6N-$C8-r$NR-s$NS-g${NG}ZZ" >> $SM/v6n_chain.log
done
echo "ZZV6NCHAIN-ENDZZ" >> $SM/v6n_chain.log
