#!/bin/bash
# r8ns_thor_smoke.sh -- refcv8 NavSim bridge on Thor CUDA, NAVTEST smoke5 ONLY (VALIDATION ONLY, never refcv8's NavSim score).
# Both EMA checkpoints; arms R8_LEGAL + R8_LEGAL_s1 forwards, PRIOR_ha0p + R8_RT derived (+ R8_RT_s1 via derive_rt_seed.py).
# NO scoring here (no PDMS is produced); NO navtrain token is touched.
# Launched as:  flock -n /home/nvidia/refcv7_post/thor_gpu.lock bash r8ns_thor_smoke.sh   (the Thor GPU-lock convention)
# Completion marker in stdout: ZZSMOKEDONE / ZZBUSY / ZZK0FAIL_<step> (opaque tokens, never grep the command text).
set -u
R=/home/nvidia/r8ns; C=$R/code; P=$C/pkg; O=$R/out; T=$C/thor
PY=/home/nvidia/venvs/tanitad-train/bin/python   # tanitad-edge has no timm (refcv8 trunk); see GROUNDWORK venv note
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy   # read-only overlay: edge scipy 1.18.0 (knots_to_navsim CubicSpline), LAST on the path
export R7_GPU_LOCK=$O/r8ns_gpu_token.lock
R7CODE=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code
INP=$R/navtest/inputs/navtest_inputs.json.gz; BANK=$R/navtest/frame_bank; LOGS=$R/navtest/navsim_logs/test
ROAD=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs/road_plane_navhard_warmup_logs.json
CFG=$R/ckpt/config.json; TOKS=$P/raw/smoke5_tokens.json
RT=$C/retime/refcv8_retime_deploy_gains_pf.json; RTSHA=3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7
RTSTAGE=$C/retime/retime_stage.py; FROZEN=$C/retime/FROZEN_GAINS_RETIME_train_pf.json
LAB="VALIDATION ONLY (Thor CUDA navtest smoke5; not refcv8's NavSim score)"
cd $O
date -Is > $O/smoke_started.txt
# --- the box must be idle: no compute app on the card, no other python of this user ---
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
pys=$(ps -u nvidia -o pid=,comm= | awk '$2 ~ /^python/' | wc -l)
echo "precheck compute_apps=$apps other_python=$pys" | tee $O/smoke_precheck.txt
if [ "$apps" != "0" ] || [ "$pys" != "0" ]; then echo ZZBUSY; exit 3; fi
for STEP in 40000 48852; do
  CK=$R/ckpt/ckpt_ema_$STEP.pt; D=$O/smoke_ema$STEP; mkdir -p $D
  case $STEP in 40000) MD5=57807456bd7169800be159ff95871d22;; 48852) MD5=b4364029b68f29a33eeb4aebbbbd002b;; esac
  # 1. the package's own K0 / KD / premises controls on CUDA (the runner's gate before any CUDA split)
  R8_TEST_CKPT=$CK R8_TEST_CONFIG=$CFG R8_TEST_DEVICE=cuda $PY $T/r8ns_cuda_controls.py $P $D/cuda_controls.json > $D/cuda_controls.log 2>&1
  K0=$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('K0_pass'))" 2>/dev/null)
  KD=$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('KD_pass'))" 2>/dev/null)
  echo "step $STEP K0=$K0 KD=$KD"
  if [ "$K0" != "True" ]; then echo "ZZK0FAIL_$STEP"; continue; fi
  DEDUP=""; [ "$KD" = "True" ] && DEDUP="--exact-dedup"
  # 2. the dev-box lock module, pointed at a Thor-local token file (run_bridge8 refuses CUDA without a held token)
  TOK=$($PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; t=gpu_lock.acquire('r8ns-smoke-$STEP',0,log=lambda m:None); print(t or 'NONE')")
  echo "step $STEP token=$TOK"
  if [ "$TOK" = "NONE" ]; then echo "ZZNOTOKEN_$STEP"; continue; fi
  # 3. the bridge, UNCHANGED, wrapped only to record torch.cuda.max_memory_allocated + wall
  $PY $T/r8ns_bridge_wrap.py $P/code -- --split navtest --arms R8_LEGAL,R8_LEGAL_s1 --inputs $INP --bank1 $BANK \
      --bank1-kind navtest --logs-root $LOGS --road-plane $ROAD --ckpt $CK --config $CFG --ckpt-md5 $MD5 --label "$LAB" \
      --device cuda --precision auto --threads 6 --out $D/bridge_navtest --derived --retime --retime-table $RT \
      --retime-table-sha256 $RTSHA --retime-stage-file $RTSTAGE --tokens-file $TOKS --gpu-lock-token $TOK $DEDUP \
      > $D/bridge.log 2>&1
  $PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; print('released', gpu_lock.release('$TOK'))"
  # 4. R8_RT at inference seed 1 (CPU, no forward)
  $PY $T/derive_rt_seed.py --pkg-code $P/code --out $D/bridge_navtest --inputs $INP --split navtest --src-arm R8_LEGAL_s1 \
      --dst-arm R8_RT_s1 --retime-table $RT --retime-table-sha256 $RTSHA --retime-stage-file $RTSTAGE --tokens-file $TOKS \
      > $D/derive_rt_s1.log 2>&1
done
# 5. the R8_RT table assertion, by content
$PY $T/rt_assert.py $FROZEN $RT $O/RT_ASSERT.json $O/smoke_ema40000/bridge_navtest $O/smoke_ema48852/bridge_navtest > $O/rt_assert.log 2>&1
date -Is > $O/smoke_finished.txt
echo ZZSMOKEDONE
