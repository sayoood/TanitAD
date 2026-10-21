#!/bin/bash
# r8ns_navoff_LAUNCH.sh -- R8-NAVFEED-DIAG (GOALS_AND_CLAIMS, registered before this arm is scored): refcv8 EMA-48,852 with the nav
# command WITHHELD (bridge arm R8_NAVOFF, nav_cmd=None = the model's nav-zero intervention) on the registered navtrain sample, Thor
# CUDA; then scored on the Thor CPU backend with the SAME driver as the selection run.  POST-HOC DIAGNOSTIC, never a bar.
# Derived from r8ns_navtrain_select_LAUNCH.sh (differences marked # NO).  Launch behind the GPU flock (blocking, -o):
#   nohup setsid flock -o /home/nvidia/refcv7_post/thor_gpu.lock bash r8ns_navoff_LAUNCH.sh > navoff.out 2>&1 < /dev/null &
# Opaque markers: ZZREFUSE_<why> / ZZNAVOFFROLLED / ZZNAVOFFSCORED.
set -u
R=/home/nvidia/r8ns; C=$R/code; P=$C/pkg; T=$C/thor; N=$R/navtrain; OUT=$R/navtrain_sel/navoff   # NO
PY=/home/nvidia/venvs/tanitad-train/bin/python
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy
export R7_GPU_LOCK=$OUT/r8ns_gpu_token.lock
R7CODE=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code
ROAD=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs/road_plane_navhard_warmup_logs.json
SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json; INP=$N/inputs/navtrain_sample_inputs.json.gz; BANK=$N/frame_bank; LOGS=$N/navsim_logs/trainval
CK=$R/ckpt/ckpt_ema_48852.pt; MD5=b4364029b68f29a33eeb4aebbbbd002b
mkdir -p $OUT; cd $OUT
$PY - "$SAMPLE" <<'EOF' || { echo ZZREFUSE_sample; exit 11; }
import hashlib, json, sys
d = json.load(open(sys.argv[1])); t = sorted(d["tokens"])
assert d["status"] == "FIXED" and len(t) == d["count"] == 2089
assert hashlib.sha256("\n".join(t).encode()).hexdigest() == d["sha256_sorted_tokens"]
EOF
for f in $INP $BANK/index.json $ROAD; do [ -s "$f" ] || { echo "ZZREFUSE_missing_$(basename $f)"; exit 12; }; done
[ "$(md5sum $CK | cut -c1-32)" = $MD5 ] || { echo ZZREFUSE_md5_48852; exit 13; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo ZZREFUSE_gpu_busy; exit 14; }   # NO: CPU scorers may run
R8_TEST_CKPT=$CK R8_TEST_CONFIG=$R/ckpt/config.json R8_TEST_DEVICE=cuda $PY $T/r8ns_cuda_controls.py $P $OUT/cuda_controls.json > $OUT/cuda_controls.log 2>&1
[ "$($PY -c "import json;print(json.load(open('$OUT/cuda_controls.json')).get('K0_pass'))")" = True ] || { echo ZZREFUSE_K0; exit 15; }
DEDUP=""; [ "$($PY -c "import json;print(json.load(open('$OUT/cuda_controls.json')).get('KD_pass'))")" = True ] && DEDUP="--exact-dedup"
TOK=$($PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; print(gpu_lock.acquire('r8ns-navoff',0,log=lambda m:None) or 'NONE')")
[ "$TOK" != NONE ] || { echo ZZREFUSE_token; exit 16; }
$PY $T/r8ns_bridge_wrap.py $P/code -- --split navtrain --arms R8_NAVOFF --inputs $INP --bank1 $BANK \
    --bank1-kind navtest --logs-root $LOGS --road-plane $ROAD --ckpt $CK --config $R/ckpt/config.json --ckpt-md5 $MD5 \
    --label "R8-NAVFEED-DIAG navtrain, EMA-48852, nav withheld (POST-HOC diagnostic, never a bar)" --device cuda --precision auto \
    --threads 6 --out $OUT/bridge --tokens-file $SAMPLE --gpu-lock-token $TOK $DEDUP > $OUT/bridge.log 2>&1
$PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; gpu_lock.release('$TOK')"
$PY - $OUT/bridge <<'EOF' || { echo ZZREFUSE_post_gate; exit 17; }
import json, os, sys
m = json.load(open(os.path.join(sys.argv[1], "seam_R8_NAVOFF.manifest.json")))
assert m.get("partial") is False and m.get("n_cv_standin_rows", 0) == 0 and m.get("n_missing", 1) == 0 and m.get("n_rows") == 2089, m
EOF
echo ZZNAVOFFROLLED
nice -n 19 ionice -c3 env CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=1 OMP_NUM_THREADS=6 python3 $C/navtrain/navtrain_cache_score.py score --profile thor \
    --sample $SAMPLE --out $OUT --label r8sel_navoff_R8_NAVOFF --seam $OUT/bridge/seam_R8_NAVOFF.npz \
    --exp $N/nsexp/r8sel_navtrain --data $N/nsdata/openscene --maps $N/nsdata/maps > $OUT/score.log 2>&1
grep -q '"status": *"PASS"' $OUT/scores/r8sel_navoff_R8_NAVOFF/r8sel_navoff_R8_NAVOFF.counts.json && echo ZZNAVOFFSCORED || echo ZZREFUSE_score
