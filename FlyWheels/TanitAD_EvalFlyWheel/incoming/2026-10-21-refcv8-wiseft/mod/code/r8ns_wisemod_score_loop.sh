#!/bin/bash
# CPU scoring loop for r8ns_wisemod_LAUNCH.sh (same driver and experiment dir as the selection run); REFUSES to score before
# PREREG_ACK_MOD.txt (the landing commit of R8-WISEFT-MOD's registration) exists.  Markers ZZMSCORED_<label> / ZZMSCOREFAIL_<label> / ZZMALLSCORED.
#   cd /home/nvidia/r8ns/navtrain_sel/wiseft && nohup setsid bash code/r8ns_wisemod_score_loop.sh > score_loop_mod.out 2>&1 < /dev/null &
set -u
R=/home/nvidia/r8ns; C=$R/code; N=$R/navtrain; OUT=$R/navtrain_sel/wiseft; SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json; W=$OUT/../wisemod.out
JOBS="wmod_PLAN:R8_LEGAL wmod_PLAN:R8_LEGAL_s1 wmod_PERC:R8_LEGAL wmod_PERC:R8_LEGAL_s1"
t0=$(date +%s)
until [ -s $OUT/PREREG_ACK_MOD.txt ]; do [ $(( $(date +%s) - t0 )) -gt 28800 ] && { echo ZZMSCORETIMEOUT_ack; exit 1; }; sleep 30; done
echo "ZZMACK $(head -c 200 $OUT/PREREG_ACK_MOD.txt)"
for J in $JOBS; do
  TAG=${J%%:*}; A=${J##*:}; LBL=r8sel_${TAG}_$A
  until [ -e $OUT/$TAG/bridge/READY_$A ]; do
    grep -qE 'ZZREFUSE_(build|gpu_busy)' $W 2>/dev/null && { echo "ZZMSCOREABORT upstream refused"; exit 1; }
    grep -qE "ZZREFUSE_.*_${TAG}(_|\$)|ZZWALLROLLED" $W 2>/dev/null && [ ! -e $OUT/$TAG/bridge/READY_$A ] && { echo "ZZMSCORESKIP_$LBL"; continue 2; }
    [ $(( $(date +%s) - t0 )) -gt 28800 ] && { echo ZZMSCORETIMEOUT; exit 1; }
    sleep 30
  done
  nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=1 OMP_NUM_THREADS=6 python3 $C/navtrain/navtrain_cache_score.py score --profile thor \
      --sample $SAMPLE --out $OUT --label $LBL --seam $OUT/$TAG/bridge/seam_$A.npz \
      --exp $N/nsexp/r8sel_navtrain --data $N/nsdata/openscene --maps $N/nsdata/maps > $OUT/score_$LBL.log 2>&1
  grep -q '"status": *"PASS"' $OUT/scores/$LBL/$LBL.counts.json 2>/dev/null && echo "ZZMSCORED_$LBL" || echo "ZZMSCOREFAIL_$LBL"
done
echo ZZMALLSCORED
