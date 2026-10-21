#!/bin/bash
# r8ns_wisemod2_LAUNCH.sh -- R8-WISEFT-MOD2 (GOALS_AND_CLAIMS, registered before any of these arms is scored): the lambda = 0.5 blend
# split of the PLAN locus (DEC, TAC) and PLAN at lambda 0.75 (build_wiseft_mod2.py, controls in ckpt_mod2/WISEFT_MOD2_BUILD.json).
# Rolls R8_LEGAL + R8_LEGAL_s1 per arm on the registered navtrain sample, Thor CUDA, behind the GPU flock (blocking, -o):
#   nohup setsid flock -o /home/nvidia/refcv7_post/thor_gpu.lock bash r8ns_wisemod2_LAUNCH.sh > wisemod2.out 2>&1 < /dev/null &
# Identical to r8ns_wiseft_LAUNCH.sh's roll() (same flags, same post-gate); scoring is r8ns_wisemod2_score_loop.sh, gated on PREREG_ACK_MOD2.txt.
set -u
R=/home/nvidia/r8ns; C=$R/code; P=$C/pkg; T=$C/thor; N=$R/navtrain; OUT=$R/navtrain_sel/wiseft
PY=/home/nvidia/venvs/tanitad-train/bin/python
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy
export R7_GPU_LOCK=$OUT/r8ns_gpu_token_mod2.lock
R7CODE=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code
ROAD=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs/road_plane_navhard_warmup_logs.json
SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json; INP=$N/inputs/navtrain_sample_inputs.json.gz; BANK=$N/frame_bank; LOGS=$N/navsim_logs/trainval
cd $OUT
grep -q '"all_pass": true' $OUT/ckpt_mod2/WISEFT_MOD2_BUILD.json || { echo ZZREFUSE_build; exit 15; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo ZZREFUSE_gpu_busy; exit 14; }
roll() {  # $1 = tag, $2 = ckpt, $3 = arms
  local TAG=$1 CK=$2 ARMS=$3 MD5 D TOK DEDUP
  MD5=$(md5sum $CK | cut -c1-32); D=$OUT/$TAG; mkdir -p $D
  R8_TEST_CKPT=$CK R8_TEST_CONFIG=$R/ckpt/config.json R8_TEST_DEVICE=cuda $PY $T/r8ns_cuda_controls.py $P $D/cuda_controls.json > $D/cuda_controls.log 2>&1
  [ "$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('K0_pass'))")" = True ] || { echo "ZZREFUSE_K0_$TAG"; return 1; }
  DEDUP=""; [ "$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('KD_pass'))")" = True ] && DEDUP="--exact-dedup"
  TOK=$($PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; print(gpu_lock.acquire('r8ns-wisemod2',0,log=lambda m:None) or 'NONE')")
  [ "$TOK" != NONE ] || { echo "ZZREFUSE_token_$TAG"; return 1; }
  $PY $T/r8ns_bridge_wrap.py $P/code -- --split navtrain --arms $ARMS --inputs $INP --bank1 $BANK \
      --bank1-kind navtest --logs-root $LOGS --road-plane $ROAD --ckpt $CK --config $R/ckpt/config.json --ckpt-md5 $MD5 \
      --label "R8-WISEFT-MOD2 navtrain, $TAG (POST-HOC attribution, never a bar)" --device cuda --precision auto \
      --threads 6 --out $D/bridge --tokens-file $SAMPLE --gpu-lock-token $TOK $DEDUP > $D/bridge.log 2>&1
  $PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; gpu_lock.release('$TOK')"
  for A in ${ARMS//,/ }; do
    $PY - $D/bridge $A <<'PYEOF' || { echo "ZZREFUSE_post_gate_${TAG}_$A"; return 1; }
import json, os, sys
m = json.load(open(os.path.join(sys.argv[1], f"seam_{sys.argv[2]}.manifest.json")))
assert m.get("partial") is False and m.get("n_cv_standin_rows", 0) == 0 and m.get("n_missing", 1) == 0 and m.get("n_rows") == 2089, m
PYEOF
    echo "ZZWROLLED_${TAG}_$A"; touch $D/bridge/READY_$A
  done
}
for G in TAC DEC PLAN75; do roll wmod2_$G $OUT/ckpt_mod2/ckpt_wiseft_mod2_$G.pt R8_LEGAL,R8_LEGAL_s1; done
echo ZZWALLROLLED
