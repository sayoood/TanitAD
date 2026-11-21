#!/bin/bash
# r8ns_wiseft_LAUNCH.sh -- R8-WISEFT + R8-CKPT-TRAJ (GOALS_AND_CLAIMS, registered before any of these arms is scored).
# Stage A (GPU, Thor CUDA, registered navtrain sample 2,089 tokens): build theta_lam = (1-lam)*theta7 + lam*theta8 for lam in
# {0, 0.25, 0.5, 0.75} (theta7 = refcv8's exact --init-from, refcv7-r101-s0 ckpt.pt md5 d5f104ee...; theta8 = EMA-48,852), roll
# R8_LEGAL + R8_LEGAL_s1 for each; then roll R8_LEGAL (seed 0) at the refcv8 EMA snapshots 5,000 / 15,000 / 30,000.  Scoring is a
# SEPARATE CPU loop (r8ns_wiseft_score_loop.sh) that scores each seam as soon as its manifest is complete.  Derived from
# r8ns_navoff_LAUNCH.sh.  Launch behind the GPU flock (blocking, -o):
#   nohup setsid flock -o /home/nvidia/refcv7_post/thor_gpu.lock bash r8ns_wiseft_LAUNCH.sh > wiseft.out 2>&1 < /dev/null &
# Opaque markers: ZZREFUSE_<why> / ZZWROLLED_<arm> / ZZWALLROLLED.
set -u
R=/home/nvidia/r8ns; C=$R/code; P=$C/pkg; T=$C/thor; N=$R/navtrain; OUT=$R/navtrain_sel/wiseft
PY=/home/nvidia/venvs/tanitad-train/bin/python
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy
export R7_GPU_LOCK=$OUT/r8ns_gpu_token.lock
R7CODE=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code
ROAD=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs/road_plane_navhard_warmup_logs.json
SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json; INP=$N/inputs/navtrain_sample_inputs.json.gz; BANK=$N/frame_bank; LOGS=$N/navsim_logs/trainval
TH7=/home/nvidia/refcv7_run/runs/refcv7-r101-s0/ckpt.pt; TH7MD5=d5f104ee54ba6b2861e38030b4f6fcf1
TH8=$R/ckpt/ckpt_ema_48852.pt; TH8MD5=b4364029b68f29a33eeb4aebbbbd002b
RUN8=/home/nvidia/refcv8_run/runs/refcv8-r101-s0
mkdir -p $OUT/ckpt; cd $OUT
# the registration gate binds SCORING (r8ns_wiseft_score_loop.sh refuses to score without PREREG_ACK.txt); rolls emit no score.
$PY - "$SAMPLE" <<'PYEOF' || { echo ZZREFUSE_sample; exit 11; }
import hashlib, json, sys
d = json.load(open(sys.argv[1])); t = sorted(d["tokens"])
assert d["status"] == "FIXED" and len(t) == d["count"] == 2089
assert hashlib.sha256("\n".join(t).encode()).hexdigest() == d["sha256_sorted_tokens"]
PYEOF
for f in $INP $BANK/index.json $ROAD $TH7 $TH8; do [ -s "$f" ] || { echo "ZZREFUSE_missing_$(basename $f)"; exit 12; }; done
[ "$(md5sum $TH7 | cut -c1-32)" = $TH7MD5 ] || { echo ZZREFUSE_md5_theta7; exit 13; }
[ "$(md5sum $TH8 | cut -c1-32)" = $TH8MD5 ] || { echo ZZREFUSE_md5_theta8; exit 13; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo ZZREFUSE_gpu_busy; exit 14; }
# ---- build (CPU; controls C0/C1 + per-lambda read-back must pass or nothing is rolled)
$PY $OUT/code/build_wiseft.py --theta7 $TH7 --theta8 $TH8 --lams 0,0.25,0.5,0.75 --out-dir $OUT/ckpt > $OUT/build.log 2>&1
grep -q '^ZZWISEBUILT' $OUT/build.log || { echo ZZREFUSE_build; exit 15; }
roll() {  # $1 = arm-set tag, $2 = ckpt, $3 = arms
  local TAG=$1 CK=$2 ARMS=$3 MD5 D TOK DEDUP
  MD5=$(md5sum $CK | cut -c1-32); D=$OUT/$TAG; mkdir -p $D
  R8_TEST_CKPT=$CK R8_TEST_CONFIG=$R/ckpt/config.json R8_TEST_DEVICE=cuda $PY $T/r8ns_cuda_controls.py $P $D/cuda_controls.json > $D/cuda_controls.log 2>&1
  [ "$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('K0_pass'))")" = True ] || { echo "ZZREFUSE_K0_$TAG"; return 1; }
  DEDUP=""; [ "$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('KD_pass'))")" = True ] && DEDUP="--exact-dedup"
  TOK=$($PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; print(gpu_lock.acquire('r8ns-wiseft',0,log=lambda m:None) or 'NONE')")
  [ "$TOK" != NONE ] || { echo "ZZREFUSE_token_$TAG"; return 1; }
  $PY $T/r8ns_bridge_wrap.py $P/code -- --split navtrain --arms $ARMS --inputs $INP --bank1 $BANK \
      --bank1-kind navtest --logs-root $LOGS --road-plane $ROAD --ckpt $CK --config $R/ckpt/config.json --ckpt-md5 $MD5 \
      --label "R8-WISEFT / R8-CKPT-TRAJ navtrain, $TAG (POST-HOC lever read, never a bar)" --device cuda --precision auto \
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
for L in 0p50 0p00 0p75 0p25; do roll wise_l$L $OUT/ckpt/ckpt_wiseft_l$L.pt R8_LEGAL,R8_LEGAL_s1; done
for S in 5000 15000 30000; do roll traj_ema$S $RUN8/ckpt_ema_$S.pt R8_LEGAL; done
echo ZZWALLROLLED
