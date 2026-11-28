#!/bin/bash
# CPU scoring loop for r8ns_wisemod2_LAUNCH.sh (same driver and experiment dir as the selection run); REFUSES to score before
# PREREG_ACK_MOD2.txt (the landing commit of R8-WISEFT-MOD2's registration) exists.  Markers ZZM2SCORED_<label> / ZZM2SCOREFAIL_<label> / ZZM2ALLSCORED.
#   cd /home/nvidia/r8ns/navtrain_sel/wiseft && nohup setsid bash code/r8ns_wisemod2_score_loop.sh > score_loop_mod2.out 2>&1 < /dev/null &
set -u
R=/home/nvidia/r8ns; C=$R/code; N=$R/navtrain; OUT=$R/navtrain_sel/wiseft; SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json; W=$OUT/../wisemod2.out
JOBS="wmod2_TAC:R8_LEGAL wmod2_TAC:R8_LEGAL_s1 wmod2_DEC:R8_LEGAL wmod2_DEC:R8_LEGAL_s1 wmod2_PLAN75:R8_LEGAL wmod2_PLAN75:R8_LEGAL_s1"
t0=$(date +%s)
until [ -s $OUT/PREREG_ACK_MOD2.txt ]; do [ $(( $(date +%s) - t0 )) -gt 28800 ] && { echo ZZM2SCORETIMEOUT_ack; exit 1; }; sleep 30; done
echo "ZZM2ACK $(head -c 200 $OUT/PREREG_ACK_MOD2.txt)"
for J in $JOBS; do
  TAG=${J%%:*}; A=${J##*:}; LBL=r8sel_${TAG}_$A
  until [ -e $OUT/$TAG/bridge/READY_$A ]; do
    grep -qE 'ZZREFUSE_(build|gpu_busy)' $W 2>/dev/null && { echo "ZZM2SCOREABORT upstream refused"; exit 1; }
    grep -qE "ZZREFUSE_.*_${TAG}(_|\$)|ZZWALLROLLED" $W 2>/dev/null && [ ! -e $OUT/$TAG/bridge/READY_$A ] && { echo "ZZM2SCORESKIP_$LBL"; continue 2; }
    [ $(( $(date +%s) - t0 )) -gt 28800 ] && { echo ZZM2SCORETIMEOUT; exit 1; }
    sleep 30
  done
  nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=1 OMP_NUM_THREADS=6 python3 $C/navtrain/navtrain_cache_score.py score --profile thor \
      --sample $SAMPLE --out $OUT --label $LBL --seam $OUT/$TAG/bridge/seam_$A.npz \
      --exp $N/nsexp/r8sel_navtrain --data $N/nsdata/openscene --maps $N/nsdata/maps > $OUT/score_$LBL.log 2>&1
  grep -q '"status": *"PASS"' $OUT/scores/$LBL/$LBL.counts.json 2>/dev/null && echo "ZZM2SCORED_$LBL" || echo "ZZM2SCOREFAIL_$LBL"
done
echo ZZM2ALLSCORED
