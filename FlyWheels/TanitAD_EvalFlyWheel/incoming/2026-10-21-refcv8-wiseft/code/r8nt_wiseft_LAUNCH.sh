#!/bin/bash
# r8nt_wiseft_LAUNCH.sh <TAG> -- R8-WISEFT STAGE 2: the ONE registered navtest read of theta_lambda* (PREREG_R8_WISEFT.json, landed
# e8911b1).  Runs ONLY if the stage-1 navtrain rule selected lambda* (the dev-box waiter passes its tag, e.g. 0p50).  Derived from
# r8nt_raw_beside_LAUNCH.sh; differences marked # WISE.  Thor CUDA rolls only; scoring = r8nt_score.py on the Thor CPU backend.
#   launch: cd /home/nvidia/r8ns/navtest_read && nohup setsid flock -o /home/nvidia/refcv7_post/thor_gpu.lock \
#             bash /home/nvidia/r8ns/navtrain_sel/wiseft/code/r8nt_wiseft_LAUNCH.sh <TAG> > wiseft_<TAG>.out 2>&1 < /dev/null &
# Opaque markers: ZZREFUSE_<why> / ZZNAVTESTDONE / ZZALLDONE.
set -u
TAG=${1:-}
case "$TAG" in 0p00) MD5=82b135a5d2209b5ec880fd8b538d000c;; 0p25) MD5=4d8b2a9323be64de23b6bc0059859cf4;;
               0p50) MD5=d72f7f575388cb32a938d2bfbbee39b5;; 0p75) MD5=dc5a3a5a428e3a6af685fa109a9dae38;;
               *) echo ZZREFUSE_tag; exit 9;; esac                                                    # WISE: builder md5s (e8911b1)
R=/home/nvidia/r8ns; C=$R/code; P=$C/pkg; T=$C/thor; X=$R/navtest_read; NT=$R/navtest; NX=$C/navtest_read
PY=/home/nvidia/venvs/tanitad-train/bin/python
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy
R7CODE=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code
R6IN=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs
ROAD=$R6IN/road_plane_navhard_warmup_logs.json; SPEED=$R6IN/speed_limits_navtest.json.gz
CK=$R/navtrain_sel/wiseft/ckpt/ckpt_wiseft_l$TAG.pt; CFG=$R/ckpt/config.json                      # WISE
RT=$C/retime/refcv8_retime_deploy_gains_pf.json; RTSHA=3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7
RTSTAGE=$C/retime/retime_stage.py; FROZEN=$C/retime/FROZEN_GAINS_RETIME_train_pf.json
INP=$NT/inputs/navtest_inputs.json.gz; BANK=$NT/frame_bank; LOGS=$NT/nsdata/openscene/navsim_logs/test; PREP=$X/prep
OUT=$X/wiseft_l$TAG; NEXP=12146; LAB="R8-WISEFT stage 2 (registered e8911b1): theta_lambda* = l$TAG on navtest, Thor CUDA"   # WISE
mkdir -p $OUT; cd $OUT
export R7_GPU_LOCK=$OUT/r8nt_gpu_token.lock
date -Is > $OUT/started.txt
[ -s $R/navtrain_sel/wiseft/PREREG_ACK.txt ] || { echo ZZREFUSE_no_prereg_ack; exit 10; }            # WISE
[ "$(md5sum $CK | cut -c1-32)" = "$MD5" ] || { echo ZZREFUSE_md5_wiseft; exit 10; }
[ "$(sha256sum $RT | cut -c1-64)" = "$RTSHA" ] || { echo ZZREFUSE_rt_table; exit 10; }
[ "$(cut -c1-64 $RT.sha256)" = "$RTSHA" ] || { echo ZZREFUSE_rt_sidecar; exit 10; }
[ "$(sha256sum $FROZEN | cut -c1-64)" = 2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80 ] || { echo ZZREFUSE_frozen; exit 10; }
[ "$(sha256sum $RTSTAGE | cut -c1-64)" = b4bc7b79aa9a0ac02a170b4e633da0de4cd6026f76923c8d2706a0b2244f08ee ] || { echo ZZREFUSE_stage; exit 10; }
for f in $INP $SPEED $ROAD $CFG $BANK/index.json; do [ -s "$f" ] || { echo "ZZREFUSE_missing_$(basename $f)"; exit 11; }; done
$PY -c "import json,sys; d=json.load(open('$PREP/G1_CHECK.json')); sys.exit(0 if d['verdict']=='PASS' and d['facts']['n_tokens_inputs']==12146 else 1)" \
  || { echo ZZREFUSE_G1; exit 11; }
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo ZZREFUSE_gpu_busy; exit 12; }
R8_TEST_CKPT=$CK R8_TEST_CONFIG=$CFG R8_TEST_DEVICE=cuda $PY $T/r8ns_cuda_controls.py $P $OUT/cuda_controls.json > $OUT/cuda_controls.log 2>&1
[ "$($PY -c "import json;print(json.load(open('$OUT/cuda_controls.json')).get('K0_pass'))")" = True ] || { echo ZZREFUSE_K0; exit 13; }
DEDUP=""; [ "$($PY -c "import json;print(json.load(open('$OUT/cuda_controls.json')).get('KD_pass'))")" = True ] && DEDUP="--exact-dedup"
TOK=$($PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; print(gpu_lock.acquire('r8nt-wiseft',0,log=lambda m:None) or 'NONE')")
[ "$TOK" != NONE ] || { echo ZZREFUSE_token; exit 14; }
# WISE: the registered bar arm at both inference seeds; PRIOR_ha0p + R8_RT (seed 0) derived as in the EMA read
$PY $T/r8ns_bridge_wrap.py $P/code -- --split navtest --arms R8_LEGAL,R8_LEGAL_s1 --inputs $INP --speed $SPEED --bank1 $BANK \
    --bank1-kind navtest --logs-root $LOGS --road-plane $ROAD --ckpt $CK --config $CFG --ckpt-md5 $MD5 --label "$LAB" \
    --device cuda --precision auto --threads 6 --out $OUT/bridge_navtest --derived --retime --retime-table $RT \
    --retime-table-sha256 $RTSHA --retime-stage-file $RTSTAGE --gpu-lock-token $TOK $DEDUP > $OUT/bridge_navtest.log 2>&1
$PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; gpu_lock.release('$TOK')"
$PY - $OUT/bridge_navtest $NEXP > $OUT/post_gate.log 2>&1 <<'EOF' || { echo ZZREFUSE_post_gate; exit 17; }
import json, os, sys
d, n = sys.argv[1], int(sys.argv[2])
bad, rep = [], {}
for arm in ("R8_LEGAL", "R8_LEGAL_s1", "R8_RT", "PRIOR_ha0p"):
    p = os.path.join(d, f"seam_{arm}.manifest.json")
    if not os.path.exists(p) or not os.path.exists(os.path.join(d, f"seam_{arm}.npz")):
        bad.append(f"{arm}: no seam"); continue
    m = json.load(open(p))
    rep[arm] = {k: m.get(k) for k in ("n_rows", "n_missing", "partial", "n_cv_standin_rows", "n_model_rows")}
    if m.get("partial") is not False or m.get("n_missing") != 0 or m.get("n_cv_standin_rows") != 0 or m.get("n_rows") != n:
        bad.append(f"{arm}: {rep[arm]}")
    if arm in ("R8_LEGAL", "R8_LEGAL_s1"):
        rep[arm]["KPR"] = m.get("KPR")
        if (m.get("KPR") or {}).get("verdict") != "PASS":
            bad.append(f"{arm}: KPR {m.get('KPR')}")
feed = {}
for arm in ("R8_LEGAL", "R8_LEGAL_s1"):
    nz = vv = rows = 0
    for ln in open(os.path.join(d, f"rows_{arm}.jsonl")):
        r = json.loads(ln)
        if r.get("source") != "refcv8":
            continue
        rows += 1
        nz += any(abs(x) > 0 for x in r["r8_nav"])
        vv += float((r.get("vmax") or {}).get("v_max_valid", 0.0)) > 0
    feed[arm] = {"rows": rows, "rows_r8_nav_nonzero": nz, "rows_v_max_valid": vv}
    if nz:
        bad.append(f"{arm}: r8_nav non-zero on {nz} rows")
    if vv:
        bad.append(f"{arm}: v_max_valid on {vv} rows (LEGAL must feed the UNKNOWN speed row)")
print("QQPOST " + json.dumps({"bad": bad, "seams": rep, "feed": feed}))
sys.exit(1 if bad else 0)
EOF
$PY $T/rt_assert.py $FROZEN $RT $OUT/RT_ASSERT.json $OUT/bridge_navtest > $OUT/rt_assert.log 2>&1
grep -q '"verdict": "PASS"' $OUT/RT_ASSERT.json || { echo ZZREFUSE_rt_assert; exit 18; }
$PY -c "
import json; m=json.load(open('$OUT/bridge_navtest/seam_R8_LEGAL.manifest.json'))
json.dump({'label': 'RESULT', 'step': int(m['model']['step']), 'weights': 'wiseft_l$TAG', 'md5': m['model']['ckpt_md5'],
           'note': 'R8-WISEFT stage 2 (registered e8911b1); scores_navtest/ is filled by r8nt_score.py'},
          open('$OUT/INDEX.json','w'), indent=1)"
date -Is > $OUT/navtest_done.txt
echo ZZNAVTESTDONE
echo ZZALLDONE
exit 0
