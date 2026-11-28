#!/bin/bash
# r8nt_navtest_read_LAUNCH.sh -- refcv8 EMA-48,852, the ONE registered navtest read (A-R8-NS1 + AMENDMENTS 1-3; PI D-R8-CKPT (b);
# D-NS8-1 = Thor).  Thor CUDA rolls only; NO scoring here (scoring = r8nt_score.py on the Thor CPU backend, afterwards).
#   usage:  MODE=SMOKE|FULL   (SMOKE = the 20 tokens of prep/smoke20_tokens.json, VALIDATION ONLY; FULL = all 12,146 navtest tokens)
#   launch: cd /home/nvidia/r8ns/navtest_read && nohup setsid flock -n -o /home/nvidia/refcv7_post/thor_gpu.lock \
#             bash /home/nvidia/r8ns/code/navtest_read/r8nt_navtest_read_LAUNCH.sh FULL > full.out 2>&1 < /dev/null &
#   (flock -o: the lock fd is CLOSED in the child, so no grandchild -- bridge, sleep, python -- can inherit and strand the lock.)
# Priority order (a killed run still yields value): 1. navtest R8_LEGAL + R8_LEGAL_s1 + R8_PRIV forwards (+ PRIOR_ha0p, R8_RT derived,
#   R8_RT_s1 via derive_rt_seed.py) -> post-gate -> RT table assertion -> ZZNAVTESTDONE; 2. warmup R8_LEGAL,R8_LEGAL_s1,R8_BLIND;
#   3. navhard two-stage R8_LEGAL,R8_LEGAL_s1.  RAW weights: NOT here (lowest priority; ckpt_48852.pt is not on Thor).
# Every gate asserts on an ARTIFACT.  Opaque markers (stdout): ZZREFUSE_<why> / ZZNAVTESTDONE / ZZWARMUPDONE / ZZNAVHARDDONE / ZZALLDONE.
set -u
MODE=${1:-}
case "$MODE" in SMOKE|FULL) ;; *) echo ZZREFUSE_mode; exit 9;; esac
R=/home/nvidia/r8ns; C=$R/code; P=$C/pkg; T=$C/thor; X=$R/navtest_read; NT=$R/navtest; NX=$C/navtest_read
PY=/home/nvidia/venvs/tanitad-train/bin/python      # tanitad-edge has no timm (GROUNDWORK B3; same venv as the navtrain selection)
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy
R7CODE=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code
R6IN=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs
ROAD=$R6IN/road_plane_navhard_warmup_logs.json; SPEED=$R6IN/speed_limits_navtest.json.gz
CK=$R/ckpt/ckpt_ema_48852.pt; MD5=b4364029b68f29a33eeb4aebbbbd002b; CFG=$R/ckpt/config.json
RT=$C/retime/refcv8_retime_deploy_gains_pf.json; RTSHA=3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7
RTSTAGE=$C/retime/retime_stage.py; FROZEN=$C/retime/FROZEN_GAINS_RETIME_train_pf.json
INP=$NT/inputs/navtest_inputs.json.gz; BANK=$NT/frame_bank; LOGS=$NT/nsdata/openscene/navsim_logs/test; PREP=$X/prep
WU_INP=$NX/inputs/warmup_navsim_agent_inputs.json; NH_INP=$NX/inputs/navhard_navsim_agent_inputs.json
WU_BANK=$R/banks/warmup_two_stage; NH_B1=$R/banks/navhard_s1; NH_B2=$R/banks/navhard_s2
if [ "$MODE" = SMOKE ]; then OUT=$X/smoke20; TF="--tokens-file $PREP/smoke20_tokens.json"; NEXP=20
  LAB="VALIDATION ONLY (Thor CUDA navtest smoke20 on the launch path; not refcv8's NavSim score)"
else OUT=$X/full; TF=""; NEXP=12146; LAB="RESULT (EMA 48,852, A-R8-NS1 navtest read, Thor CUDA)"; fi
mkdir -p $OUT; cd $OUT
export R7_GPU_LOCK=$OUT/r8nt_gpu_token.lock
date -Is > $OUT/started.txt
# ---- G0: checkpoint md5; RETIME deploy table, its sidecar, the frozen source table, the stage and derive_rt_seed.py by content ----
[ "$(md5sum $CK | cut -c1-32)" = "$MD5" ] || { echo ZZREFUSE_md5_48852; exit 10; }
[ "$(sha256sum $RT | cut -c1-64)" = "$RTSHA" ] || { echo ZZREFUSE_rt_table; exit 10; }
[ "$(cut -c1-64 $RT.sha256)" = "$RTSHA" ] || { echo ZZREFUSE_rt_sidecar; exit 10; }
[ "$(sha256sum $FROZEN | cut -c1-64)" = 2aacc9ff32d3d2a5a18dda2d72ab5ce6c42d1c1268e9186912c5151bbe4d6d80 ] || { echo ZZREFUSE_frozen; exit 10; }
[ "$(sha256sum $RTSTAGE | cut -c1-64)" = b4bc7b79aa9a0ac02a170b4e633da0de4cd6026f76923c8d2706a0b2244f08ee ] || { echo ZZREFUSE_stage; exit 10; }
[ "$(sha256sum $T/derive_rt_seed.py | cut -c1-64)" = bebdc5c1ab41b775a60c80d4d79510bc7af6f4b4d12c374526985408e260e278 ] || { echo ZZREFUSE_derive; exit 10; }
for f in $INP $SPEED $ROAD $CFG $BANK/index.json; do [ -s "$f" ] || { echo "ZZREFUSE_missing_$(basename $f)"; exit 11; }; done
# ---- G1: the artifact of r8nt_prep.py (12,146 tokens, every token in the bank with 4 history rows, every shard valid, logs present) ----
$PY -c "import json,sys; d=json.load(open('$PREP/G1_CHECK.json')); sys.exit(0 if d['verdict']=='PASS' and d['facts']['n_tokens_inputs']==12146 else 1)" \
  || { echo ZZREFUSE_G1; exit 11; }
# ---- G2: the box is idle (we hold the flock by construction: flock -n refused otherwise) ----
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo ZZREFUSE_gpu_busy; exit 12; }
[ "$(ps -u nvidia -o comm= | awk '$1 ~ /^python/' | wc -l)" = 0 ] || { echo ZZREFUSE_python_busy; exit 12; }
# ---- K0 / KD / premises on CUDA (the package's own test functions, unchanged) ----
R8_TEST_CKPT=$CK R8_TEST_CONFIG=$CFG R8_TEST_DEVICE=cuda $PY $T/r8ns_cuda_controls.py $P $OUT/cuda_controls.json > $OUT/cuda_controls.log 2>&1
[ "$($PY -c "import json;print(json.load(open('$OUT/cuda_controls.json')).get('K0_pass'))")" = True ] || { echo ZZREFUSE_K0; exit 13; }
DEDUP=""; [ "$($PY -c "import json;print(json.load(open('$OUT/cuda_controls.json')).get('KD_pass'))")" = True ] && DEDUP="--exact-dedup"
TOK=$($PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; print(gpu_lock.acquire('r8nt-navtest',0,log=lambda m:None) or 'NONE')")
[ "$TOK" != NONE ] || { echo ZZREFUSE_token; exit 14; }
# ---- 1. navtest: 3 forward arms (R8_PRIV is DIAGNOSTIC ONLY) + PRIOR_ha0p + R8_RT (seed 0) derived ----
$PY $T/r8ns_bridge_wrap.py $P/code -- --split navtest --arms R8_LEGAL,R8_LEGAL_s1,R8_PRIV --inputs $INP --speed $SPEED --bank1 $BANK \
    --bank1-kind navtest --logs-root $LOGS --road-plane $ROAD --ckpt $CK --config $CFG --ckpt-md5 $MD5 --label "$LAB" \
    --device cuda --precision auto --threads 6 --out $OUT/bridge_navtest --derived --retime --retime-table $RT \
    --retime-table-sha256 $RTSHA --retime-stage-file $RTSTAGE $TF --gpu-lock-token $TOK $DEDUP > $OUT/bridge_navtest.log 2>&1
$PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; gpu_lock.release('$TOK')"
# R8_RT at inference seed 1 (no forward; the accepted derive_rt_seed.py)
$PY $T/derive_rt_seed.py --pkg-code $P/code --out $OUT/bridge_navtest --inputs $INP --split navtest --src-arm R8_LEGAL_s1 \
    --dst-arm R8_RT_s1 --retime-table $RT --retime-table-sha256 $RTSHA --retime-stage-file $RTSTAGE \
    $( [ "$MODE" = SMOKE ] && echo "--tokens-file $PREP/smoke20_tokens.json" ) > $OUT/derive_rt_s1.log 2>&1
# ---- post-gate: every seam complete, ZERO stand-ins, KPR PASS on the forward arms, the LEGAL feed as registered ----
$PY - $OUT/bridge_navtest $NEXP > $OUT/post_gate.log 2>&1 <<'EOF' || { echo ZZREFUSE_post_gate; exit 17; }
import json, os, sys
d, n = sys.argv[1], int(sys.argv[2])
bad, rep = [], {}
for arm in ("R8_LEGAL", "R8_LEGAL_s1", "R8_PRIV", "R8_RT", "R8_RT_s1", "PRIOR_ha0p"):
    p = os.path.join(d, f"seam_{arm}.manifest.json")
    if not os.path.exists(p) or not os.path.exists(os.path.join(d, f"seam_{arm}.npz")):
        bad.append(f"{arm}: no seam"); continue
    m = json.load(open(p))
    rep[arm] = {k: m.get(k) for k in ("n_rows", "n_missing", "partial", "n_cv_standin_rows", "n_model_rows")}
    if m.get("partial") is not False or m.get("n_missing") != 0 or m.get("n_cv_standin_rows") != 0 or m.get("n_rows") != n:
        bad.append(f"{arm}: {rep[arm]}")
    if arm in ("R8_LEGAL", "R8_LEGAL_s1", "R8_PRIV"):
        rep[arm]["KPR"] = m.get("KPR")
        if (m.get("KPR") or {}).get("verdict") != "PASS":
            bad.append(f"{arm}: KPR {m.get('KPR')}")
# the registered LEGAL feed, read back from the rows (Amendment 1): r8_nav all zero on every row; LEGAL arms v_max_valid 0
feed = {}
for arm in ("R8_LEGAL", "R8_LEGAL_s1", "R8_PRIV"):
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
    if arm != "R8_PRIV" and vv:
        bad.append(f"{arm}: v_max_valid on {vv} rows (LEGAL must feed the UNKNOWN speed row)")
print("QQPOST " + json.dumps({"bad": bad, "seams": rep, "feed": feed}))
sys.exit(1 if bad else 0)
EOF
$PY $T/rt_assert.py $FROZEN $RT $OUT/RT_ASSERT.json $OUT/bridge_navtest > $OUT/rt_assert.log 2>&1
grep -q '"verdict": "PASS"' $OUT/RT_ASSERT.json || { echo ZZREFUSE_rt_assert; exit 18; }
# INDEX.json in the shape navsim_bars8.load_run reads (label / step / weights / md5); step read back from the bar arm's manifest
$PY -c "
import json; m=json.load(open('$OUT/bridge_navtest/seam_R8_LEGAL.manifest.json'))
lab = 'VALIDATION ONLY (smoke20)' if '$MODE' == 'SMOKE' else 'RESULT'
json.dump({'label': lab, 'step': int(m['model']['step']), 'weights': 'ema', 'md5': m['model']['ckpt_md5'],
           'note': 'written by r8nt_navtest_read_LAUNCH.sh after the post-gate; scores_navtest/ is filled by r8nt_score.py'},
          open('$OUT/INDEX.json','w'), indent=1)"
date -Is > $OUT/navtest_done.txt
echo ZZNAVTESTDONE
[ "$MODE" = SMOKE ] && [ "${SMOKE_ALL_SPLITS:-0}" != 1 ] && { echo ZZALLDONE; exit 0; }
# ---- 2. warmup (N3): R8_LEGAL, R8_LEGAL_s1, R8_BLIND (the deliberate regression that must FAIL N3) ----
# ---- 3. navhard two-stage (N2): R8_LEGAL, R8_LEGAL_s1 ----
for SPL in warmup navhard; do
  case $SPL in
    warmup)  SPLIT=warmup_two_stage;  ARMS=R8_LEGAL,R8_LEGAL_s1,R8_BLIND; SINP=$WU_INP; BANKS="--bank2 $WU_BANK";;
    navhard) SPLIT=navhard_two_stage; ARMS=R8_LEGAL,R8_LEGAL_s1;          SINP=$NH_INP; BANKS="--bank2 $NH_B2 --bank1 $NH_B1";;
  esac
  [ -s "$SINP" ] || { echo "ZZSKIP_${SPL}_no_inputs"; continue; }
  LIM=""; [ "$MODE" = SMOKE ] && LIM="--limit 10"
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "ZZREFUSE_gpu_busy_$SPL"; exit 12; }
  TOK=$($PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; print(gpu_lock.acquire('r8nt-$SPL',0,log=lambda m:None) or 'NONE')")
  [ "$TOK" != NONE ] || { echo "ZZREFUSE_token_$SPL"; exit 14; }
  $PY $T/r8ns_bridge_wrap.py $P/code -- --split $SPLIT --arms $ARMS --inputs $SINP $BANKS --logs-root $LOGS --road-plane $ROAD --ckpt $CK \
      --config $CFG --ckpt-md5 $MD5 --label "$LAB" --device cuda --precision auto --threads 6 --out $OUT/bridge_$SPL --derived \
      $LIM --gpu-lock-token $TOK $DEDUP > $OUT/bridge_$SPL.log 2>&1
  $PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; gpu_lock.release('$TOK')"
  $PY - $OUT/bridge_$SPL $ARMS > $OUT/post_gate_$SPL.log 2>&1 <<'EOF'
import json, os, sys
d, arms = sys.argv[1], sys.argv[2].split(",") + ["PRIOR_ha0p"]
rep = {}
for arm in arms:
    p = os.path.join(d, f"seam_{arm}.manifest.json")
    m = json.load(open(p)) if os.path.exists(p) else {}
    rep[arm] = {k: m.get(k) for k in ("n_expected", "n_rows", "n_missing", "partial", "n_cv_standin_rows", "n_model_rows")}
    rep[arm]["KPR"] = (m.get("KPR") or {}).get("verdict")
print("QQPOST " + json.dumps(rep))
EOF
  date -Is > $OUT/${SPL}_done.txt
  [ $SPL = warmup ] && echo ZZWARMUPDONE || echo ZZNAVHARDDONE
done
echo ZZALLDONE
