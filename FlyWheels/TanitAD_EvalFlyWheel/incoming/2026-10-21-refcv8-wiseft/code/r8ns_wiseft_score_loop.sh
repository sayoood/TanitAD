#!/bin/bash
# CPU scoring loop for r8ns_wiseft_LAUNCH.sh: scores each seam (in roll order) as soon as its READY_<arm> flag exists, one at a time,
# nice 19, with the SAME driver and experiment dir as the D-R8-CKPT selection run.  Markers ZZWSCORED_<label> / ZZWSCOREFAIL_<label> /
# ZZWALLSCORED.  Launch: nohup setsid bash r8ns_wiseft_score_loop.sh > score_loop.out 2>&1 < /dev/null &
set -u
R=/home/nvidia/r8ns; C=$R/code; N=$R/navtrain; OUT=$R/navtrain_sel/wiseft; SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json
JOBS="wise_l0p50:R8_LEGAL wise_l0p50:R8_LEGAL_s1 wise_l0p00:R8_LEGAL wise_l0p00:R8_LEGAL_s1 wise_l0p75:R8_LEGAL wise_l0p75:R8_LEGAL_s1 wise_l0p25:R8_LEGAL wise_l0p25:R8_LEGAL_s1 traj_ema5000:R8_LEGAL traj_ema15000:R8_LEGAL traj_ema30000:R8_LEGAL"
t0=$(date +%s); W=$OUT/../wiseft.out
# REGISTRATION GATE: no seam is scored before the reading rule has landed (PREREG_ACK.txt carries the landing commit).
until [ -s $OUT/PREREG_ACK.txt ]; do [ $(( $(date +%s) - t0 )) -gt 21600 ] && { echo ZZWSCORETIMEOUT_ack; exit 1; }; sleep 30; done
echo "ZZWACK $(cat $OUT/PREREG_ACK.txt | head -c 200)"
for J in $JOBS; do
  TAG=${J%%:*}; A=${J##*:}; LBL=r8sel_${TAG}_$A
  until [ -e $OUT/$TAG/bridge/READY_$A ]; do
    grep -qE 'ZZREFUSE_(sample|missing|md5|gpu_busy|build)' $W 2>/dev/null && { echo "ZZWSCOREABORT upstream refused"; exit 1; }
    grep -qE "ZZREFUSE_.*_${TAG}(_|\$)|ZZWALLROLLED" $W 2>/dev/null && [ ! -e $OUT/$TAG/bridge/READY_$A ] && { echo "ZZWSCORESKIP_$LBL"; continue 2; }
    [ $(( $(date +%s) - t0 )) -gt 21600 ] && { echo ZZWSCORETIMEOUT; exit 1; }
    sleep 30
  done
  nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=1 OMP_NUM_THREADS=6 python3 $C/navtrain/navtrain_cache_score.py score --profile thor \
      --sample $SAMPLE --out $OUT --label $LBL --seam $OUT/$TAG/bridge/seam_$A.npz \
      --exp $N/nsexp/r8sel_navtrain --data $N/nsdata/openscene --maps $N/nsdata/maps > $OUT/score_$LBL.log 2>&1
  grep -q '"status": *"PASS"' $OUT/scores/$LBL/$LBL.counts.json 2>/dev/null && echo "ZZWSCORED_$LBL" || echo "ZZWSCOREFAIL_$LBL"
done
echo ZZWALLSCORED
