#!/bin/bash
# Thor: completes the pre-registered v6r / v6n arms after the day v3raw (rectified virtual views) turned out to hold only
# 20 of 96 frames. (1) night v6n: waits for the running rectification to finish, then refine -> score -> consensus ->
# render -> GT score; (2) day: waits for all 96 v3raw frames (sam3map_extract_v3.py filling 020-095), then v6r and v6n.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
VIEWS=CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT
PP=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map
LOG=$SM/v6rn_rest.log

post() {   # $1 c8, $2 tag (v6r | v6n)
  C8=$1; TG=$2; SUF=${TG#v6}
  cd $SM
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS SAM3MAP_TAG=$TG PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > refine_${TG}_$C8.log 2>&1
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_$TG > $SM/score_${TG}_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_consensus.py $C8 $SM/${C8}_$TG $SM/${C8}_v61$SUF > $SM/consensus_v61${SUF}_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=6 $PY sam3map_render_v5.py $C8 $SM/${C8}_v61$SUF $SM/render5_${C8}_v61$SUF > $SM/render5_${C8}_v61$SUF.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v61$SUF/worldmap.npz SAM3MAP_SCORE_TAG=gt OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_v61$SUF > $SM/score_v61${SUF}_gt_$C8.log 2>&1
  NR=$(ls $SM/${C8}_${TG}raw/[0-9][0-9][0-9].npz 2>/dev/null | wc -l); NF=$(ls $SM/${C8}_$TG/[0-9][0-9][0-9].npz 2>/dev/null | wc -l)
  NS=$(stat -c %s $SM/sam3map_score_${C8}_$TG.json 2>/dev/null || echo 0); NG=$(stat -c %s $SM/sam3map_score_${C8}_v61${SUF}_gt.json 2>/dev/null || echo 0)
  echo "ZZ$(echo $TG | tr a-z A-Z)-$C8-raw$NR-ref$NF-s$NS-g${NG}ZZ" >> $LOG
}

# ---- night v6n
until [ -s $SM/rectified_road_73495082f98b_v6nraw.json ]; do sleep 20; done
post 73495082f98b v6n

# ---- day: all 96 rectified virtual-view frames first
until [ "$(ls $SM/4fbd97b6a4b7_v3raw/[0-9][0-9][0-9].npz 2>/dev/null | wc -l)" -ge 96 ] && grep -aq "ZZSAM3MAP-DONEZZ\|Traceback" $SM/v3raw_day_fill.log 2>/dev/null; do sleep 30; done
if grep -aq Traceback $SM/v3raw_day_fill.log; then echo "ZZDAY-V3RAW-FILL-FAILEDZZ" >> $LOG; exit 1; fi
cd $SM/eval
env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS OMP_NUM_THREADS=6 $PY sam3map_rectified_road.py 4fbd97b6a4b7 $SM/4fbd97b6a4b7_v6raw $SM/4fbd97b6a4b7_v3raw $SM/4fbd97b6a4b7_v6rraw > $SM/rectified_road_4fbd97b6a4b7.log 2>&1
post 4fbd97b6a4b7 v6r
cd $SM/eval
env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS RECT_ALL=1 DROP_TELE=1 OMP_NUM_THREADS=6 $PY sam3map_rectified_road.py 4fbd97b6a4b7 $SM/4fbd97b6a4b7_v6raw $SM/4fbd97b6a4b7_v3raw $SM/4fbd97b6a4b7_v6nraw > $SM/rectified_all_4fbd97b6a4b7.log 2>&1
post 4fbd97b6a4b7 v6n
echo "ZZV6RN-REST-ENDZZ" >> $LOG
