#!/bin/bash
# Thor: v6sz = crosswalk stripes with SAM3 zoomed crops (sam3map_xwalk_stripes_zoom.py) -> front-only -> refine -> consensus
# (NO_PROMOTE) -> fields renderer v5m (d0 settings) -> compose with map r's options (zr) -> LiDAR scores, PI checks, robust A2,
# crosswalk image contrast -> videos. Night first, day second.
#
# PRE-REGISTERED 2026-09-14 before the first run. zr WORKS if, on BOTH clips (all 7 cameras), against the delivered map r:
#   crosswalk bar share of crossing pixels >= 1.3 x r's (night 0.087 -> >= 0.113, day 0.052 -> >= 0.068)
#   AND crosswalk bar/gap image top-hat >= r's - 0.05 (night >= 1.467, day >= 2.337)       [xwalk_reproj_contrast.py, same pixels]
#   AND noise fragments <= r's + 5 per 1000 m^2 (night <= 30.4, day <= 32.8)
#   AND LiDAR curb recall >= r's - 0.02 (night >= 0.782, day >= 0.769).
# Anything else is a FAIL, reported as such.
SM=/home/nvidia/sam3map
PY=/home/nvidia/venvs/tanitad-edge/bin/python
PP=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map
VIEWS=CAM_FW,CAM_CL,CAM_CR,CAM_RL,CAM_RR,CAM_RT,CAM_FT
LOG=$SM/v6sz_chain.log
COMPOSE="LINE_EDGE_GAP=1 LINE_MIN_LEN_M=1.0 WLK_ISLAND_M2=3 EDGE_OBS=near XWALK=dirclose XWALK_LEN=11"
for PAIR in 73495082f98b:86 4fbd97b6a4b7:41; do
  C8=${PAIR%%:*}; J=${PAIR##*:}
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 ZOOM_MAX_M=20 \
      $PY sam3map_xwalk_stripes_zoom.py $C8 $SM/${C8}_v6raw $SM/${C8}_v6szraw $SM/render5_${C8}_v65ma > $SM/xwalk_stripes_zoom_$C8.log 2>&1
  NS=$(ls $SM/${C8}_v6szraw/[0-9][0-9][0-9].npz 2>/dev/null | wc -l)
  [ "$NS" -eq 96 ] || { echo "ZZV6SZ-$C8-STRIPES-INCOMPLETE-$NS""ZZ" >> $LOG; continue; }
  env SAM3MAP_ROOT=$SM/native7 OMP_NUM_THREADS=4 $PY make_front_only.py $C8 $SM/${C8}_v6szraw $SM/${C8}_v6szfraw > $SM/front_only_v6sz_$C8.log 2>&1
  cd $SM
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=$VIEWS SAM3MAP_TAG=v6sz PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > refine_v6sz_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=CAM_FW SAM3MAP_TAG=v6szf PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > refine_v6szf_$C8.log 2>&1
  cd $SM/eval
  for ARM in a f; do
    TG=v6sz$([ $ARM = f ] && echo f); OUTN=v61sz$([ $ARM = f ] && echo f)
    env SAM3MAP_ROOT=$SM/native7 CONSENSUS_NO_PROMOTE=1 OMP_NUM_THREADS=5 $PY sam3map_consensus.py $C8 $SM/${C8}_$TG $SM/${C8}_$OUTN > $SM/consensus_${OUTN}_$C8.log 2>&1
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 FIELDS=1 OMP_NUM_THREADS=5 \
        $PY sam3map_render_v5m.py $C8 $SM/${C8}_$OUTN $SM/render5_${C8}_v65z$ARM > $SM/render5_${C8}_v65z$ARM.log 2>&1
    env $COMPOSE $PY compose.py $SM/render5_${C8}_v65z$ARM $SM/render5_${C8}_v65zr$ARM > $SM/compose_v65zr${ARM}_$C8.log 2>&1
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_v65zr$ARM/worldmap.npz SAM3MAP_SCORE_TAG=gt_v65zr OMP_NUM_THREADS=5 $PY thor_run.py score $C8 $SM/${C8}_$OUTN > $SM/score_v65zr${ARM}_$C8.log 2>&1
  done
  VARS="compose65r_all=$SM/render5_${C8}_v65ra compose65r_front=$SM/render5_${C8}_v65rf zoom65zr_all=$SM/render5_${C8}_v65zra zoom65zr_front=$SM/render5_${C8}_v65zrf"
  env OMP_NUM_THREADS=5 $PY pi_checks.py $C8 $SM/${C8}_v61s $SM/pi_checks_${C8}_zr.json $J -5,40,-15,15 $VARS > $SM/pi_checks_${C8}_zr.log 2>&1
  env OMP_NUM_THREADS=5 $PY a2_robust2.py $C8 $SM/${C8}_v61s $VARS > $SM/a2_robust2_${C8}_zr.log 2>&1
  env OMP_NUM_THREADS=5 $PY xwalk_reproj_contrast.py $C8 $SM/${C8}_v61s $SM/render5_${C8}_v65ma "d0 delivered=$SM/render5_${C8}_v65d0a" "r delivered=$SM/render5_${C8}_v65ra" "zr zoom stripes=$SM/render5_${C8}_v65zra" > $SM/xrc_zr_$C8.log 2>&1
  cd $SM && ./reproj_video2.sh $C8 v65zr v61szf v61sz $SM/pi_checks_${C8}_zr.json zoom65zr_front zoom65zr_all 0
  NM=$(ls $SM/render5_${C8}_v65zr[af]/worldmap.npz 2>/dev/null | wc -l)
  echo "ZZV6SZ-$C8-stripes$NS-maps$NM-checks$(grep -c ZZPICHECKS $SM/pi_checks_${C8}_zr.log)-xrc$(grep -c ZZXRC $SM/xrc_zr_$C8.log)ZZ" >> $LOG
done
echo "ZZV6SZ-ENDZZ" >> $LOG
