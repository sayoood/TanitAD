#!/bin/bash
# Thor: SAM3 speed approval chain, part 2 (SPEC_sam3_speed_prereg.md, same bars). Starts after part 1's GPU probes.
#   spdF2a = spdF2 + ASYNC=2 (CPU post-processing in 2 threads)       -> must be bit-identical to spdF2
#   spdF3  = spdF2 + fp16 autocast on the image backbone               -> numerical bars vs spd0
#   spdF3a = spdF3 + ASYNC=2                                          -> must be bit-identical to spdF3
# Correctness only (downstream of part 1 may still be running on the CPU); timings come from the clean pace run.
SM=/home/nvidia/sam3map; PY=/home/nvidia/venvs/tanitad-edge/bin/python
PP=/home/nvidia/sam3vendor:/home/nvidia/sam3paint:/home/nvidia/sam3map
LOG=$SM/spd_chain.log
CLIPS="73495082f98b 4fbd97b6a4b7"
until grep -qs "ZZFP8WATCH" $LOG; do
  [ "$(ps -eo args | grep -c "[s]pd_fp8_watch.sh")" -eq 0 ] && break
  sleep 30
done
echo "part2 start $(date -Is)" >> $LOG

extract() {
  local T=$1; shift
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 "$@" $PY sam3map_front_fast.py $T $CLIPS > $SM/front_fast_$T.log 2>&1
  echo "ZZSPDX-$T-$(ls $SM/73495082f98b_${T}raw/ 2>/dev/null | grep -c npz)-$(ls $SM/4fbd97b6a4b7_${T}raw/ 2>/dev/null | grep -c npz)-$(grep -c ZZFRONTFAST $SM/front_fast_$T.log)ZZ" >> $LOG
}
downstream() {
  local T=$1 C8=$2
  cd $SM
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_VIEWS=CAM_FW SAM3MAP_TAG=$T PYTHONPATH=$PP HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 $PY sam3map_refine_v6.py $C8 > $SM/refine_${T}_$C8.log 2>&1
  cd $SM/eval
  env SAM3MAP_ROOT=$SM/native7 CONSENSUS_NO_PROMOTE=1 OMP_NUM_THREADS=4 $PY sam3map_consensus.py $C8 $SM/${C8}_$T $SM/${C8}_${T}c > $SM/consensus_${T}_$C8.log 2>&1
  env SAM3MAP_ROOT=$SM/native7 SAM3MAP_COMPOSITE=majority_v65 SURFACE_MODE=5 PAINT_SHARE=0.2 PAINT_RULE=p95 CROSSWALK_STRIPES=1 NO_FRAMES=1 FIELDS=1 OMP_NUM_THREADS=4 \
      $PY sam3map_render_v5m.py $C8 $SM/${C8}_${T}c $SM/render5_${C8}_${T}m > $SM/render5_${C8}_${T}m.log 2>&1
  env LINE_EDGE_GAP=1 LINE_MIN_LEN_M=1.0 WLK_ISLAND_M2=3 EDGE_OBS=near XWALK=dirclose XWALK_LEN=11 OMP_NUM_THREADS=4 \
      $PY compose.py $SM/render5_${C8}_${T}m $SM/render5_${C8}_${T}r > $SM/compose_${T}_$C8.log 2>&1
  echo "ZZSPDD-$T-$C8-ref$(grep -c ZZREFINE6-DONEZZ $SM/refine_${T}_$C8.log)-con$(grep -c ZZCONSENSUS-DONEZZ $SM/consensus_${T}_$C8.log)-map$(ls $SM/render5_${C8}_${T}r/ 2>/dev/null | grep -c worldmap.npz)ZZ" >> $LOG
}

extract spdF2a MODE=fast HALF=fp16all BATCH=19 ASYNC=2
extract spdF3 MODE=fast HALF=fp16all BATCH=19 BB_HALF=fp16
extract spdF3a MODE=fast HALF=fp16all BATCH=19 BB_HALF=fp16 ASYNC=2
# part 1 must have finished its downstream before part 2's refine runs (both use the GPU briefly; and pi_checks reads spd0r)
until grep -qs "ZZSPD1-ENDZZ" $LOG; do
  [ "$(ps -eo args | grep -c "[s]pd_chain1.sh")" -eq 0 ] && break
  sleep 30
done
for T in spdF2a spdF3 spdF3a; do
  for C8 in $CLIPS; do downstream $T $C8; done
done
cd $SM/eval
for C8 in $CLIPS; do
  J=$([ $C8 = 73495082f98b ] && echo 86 || echo 41)
  env OMP_NUM_THREADS=4 $PY pi_checks.py $C8 $SM/${C8}_v61s $SM/pi_checks_${C8}_spd2.json $J -5,40,-15,15 spd0=$SM/render5_${C8}_spd0r spdF2a=$SM/render5_${C8}_spdF2ar spdF3=$SM/render5_${C8}_spdF3r spdF3a=$SM/render5_${C8}_spdF3ar > $SM/pi_checks_${C8}_spd2.log 2>&1
  echo "ZZSPDC2-$C8-$(grep -c ZZPICHECKS $SM/pi_checks_${C8}_spd2.log)ZZ" >> $LOG
done
for T in spdF3; do
  for C8 in $CLIPS; do
    env SAM3MAP_ROOT=$SM/native7 SAM3MAP_WORLDMAP=$SM/render5_${C8}_${T}r/worldmap.npz SAM3MAP_SCORE_TAG=gt_$T OMP_NUM_THREADS=4 $PY thor_run.py score $C8 $SM/${C8}_${T}c > $SM/score_${T}_$C8.log 2>&1
  done
done
echo "ZZSPD2-ENDZZ $(date -Is)" >> $LOG
