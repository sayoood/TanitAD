#!/bin/bash
# r8ns_navtrain_select_LAUNCH.sh -- D-R8-CKPT-RULE navtrain selection rolls on Thor.  ⛔ WRITTEN 2026-10-20, NOT EXECUTED.
# Produces the R8_RT seams (seed 0 + seed 1) for EMA-40k and EMA-48,852 on the registered navtrain sample, then hands them to the
# (NOT YET BUILT) navtrain scorer.  Every prerequisite is an ARTIFACT gate; any missing one REFUSES before the GPU is touched.
#   launch:  flock -n /home/nvidia/refcv7_post/thor_gpu.lock bash r8ns_navtrain_select_LAUNCH.sh  > select.out 2>&1
# Opaque markers: ZZREFUSE_<why> / ZZROLLSDONE.
set -u
R=/home/nvidia/r8ns; C=$R/code; P=$C/pkg; T=$C/thor; N=$R/navtrain; OUT=$R/navtrain_sel
PY=/home/nvidia/venvs/tanitad-train/bin/python     # see GROUNDWORK.md blocker B3 (edge has no timm); needs PI/MM acceptance
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy
export R7_GPU_LOCK=$OUT/r8ns_gpu_token.lock
R7CODE=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code
ROAD=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs/road_plane_navhard_warmup_logs.json
RT=$C/retime/refcv8_retime_deploy_gains_pf.json; RTSHA=3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7
RTSTAGE=$C/retime/retime_stage.py; FROZEN=$C/retime/FROZEN_GAINS_RETIME_train_pf.json
SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json                 # status must read FIXED (GROUNDWORK.md section 2)
INP=$N/inputs/navtrain_sample_inputs.json.gz           # B5a: navtrain port of W3 export_navtest_inputs.py (NOT BUILT)
BANK=$N/frame_bank                                     # B5b: navtrain port of build_navtest416.py, BankNavtest416 layout (NOT BUILT)
LOGS=$N/navsim_logs/trainval                           # the sample logs' OpenScene trainval pickles (download B1)
mkdir -p $OUT
# ---- G0: registered rule acknowledged + amendment filed (B2) -- the MM writes the GOALS_AND_CLAIMS blob sha here ----
[ -s $OUT/AMENDMENT_ACK.txt ] || { echo ZZREFUSE_no_amendment_ack; exit 10; }
# ---- G1: the sample is FIXED, >= 1,000 tokens, and its recorded sha256 re-derives ----
$PY - "$SAMPLE" <<'EOF' || { echo ZZREFUSE_sample; exit 11; }
import hashlib, json, sys
d = json.load(open(sys.argv[1])); t = sorted(d["tokens"])
h = hashlib.sha256("\n".join(t).encode()).hexdigest()
assert d["status"] == "FIXED", d["status"]
assert len(t) == d["count"] >= 1000 and len(set(t)) == len(t), (len(t), d["count"])
assert h == d["sha256_sorted_tokens"], (h, d["sha256_sorted_tokens"])
EOF
# ---- G2: inputs + bank + logs exist; checkpoints md5; retime table by content ----
for f in $INP $BANK/index.json $ROAD $RT $RT.sha256 $RTSTAGE $FROZEN; do [ -s "$f" ] || { echo "ZZREFUSE_missing_$(basename $f)"; exit 12; }; done
[ "$(ls $LOGS/*.pkl 2>/dev/null | wc -l)" -gt 0 ] || { echo ZZREFUSE_no_logs; exit 12; }
[ "$(md5sum $R/ckpt/ckpt_ema_40000.pt | cut -c1-32)" = 57807456bd7169800be159ff95871d22 ] || { echo ZZREFUSE_md5_40000; exit 13; }
[ "$(md5sum $R/ckpt/ckpt_ema_48852.pt | cut -c1-32)" = b4364029b68f29a33eeb4aebbbbd002b ] || { echo ZZREFUSE_md5_48852; exit 13; }
# ---- G3: the box is idle (Thor GPU sharing is net-negative; never add load to a live job) ----
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo ZZREFUSE_gpu_busy; exit 14; }
[ "$(ps -u nvidia -o comm= | awk '$1 ~ /^python/' | wc -l)" = 0 ] || { echo ZZREFUSE_python_busy; exit 14; }
for STEP in 40000 48852; do
  case $STEP in 40000) MD5=57807456bd7169800be159ff95871d22;; 48852) MD5=b4364029b68f29a33eeb4aebbbbd002b;; esac
  CK=$R/ckpt/ckpt_ema_$STEP.pt; D=$OUT/ema$STEP; mkdir -p $D
  R8_TEST_CKPT=$CK R8_TEST_CONFIG=$R/ckpt/config.json R8_TEST_DEVICE=cuda $PY $T/r8ns_cuda_controls.py $P $D/cuda_controls.json > $D/cuda_controls.log 2>&1
  [ "$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('K0_pass'))")" = True ] || { echo "ZZREFUSE_K0_$STEP"; exit 15; }
  DEDUP=""; [ "$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('KD_pass'))")" = True ] && DEDUP="--exact-dedup"
  TOK=$($PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; print(gpu_lock.acquire('r8ns-navtrain-$STEP',0,log=lambda m:None) or 'NONE')")
  [ "$TOK" != NONE ] || { echo "ZZREFUSE_token_$STEP"; exit 16; }
  $PY $T/r8ns_bridge_wrap.py $P/code -- --split navtrain --arms R8_LEGAL,R8_LEGAL_s1 --inputs $INP --bank1 $BANK \
      --bank1-kind navtest --logs-root $LOGS --road-plane $ROAD --ckpt $CK --config $R/ckpt/config.json --ckpt-md5 $MD5 \
      --label "D-R8-CKPT-RULE navtrain selection, EMA-$STEP (registered 2026-10-20)" --device cuda --precision auto \
      --threads 6 --out $D/bridge --derived --retime --retime-table $RT --retime-table-sha256 $RTSHA \
      --retime-stage-file $RTSTAGE --tokens-file $SAMPLE --gpu-lock-token $TOK $DEDUP > $D/bridge.log 2>&1
  $PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; gpu_lock.release('$TOK')"
  $PY $T/derive_rt_seed.py --pkg-code $P/code --out $D/bridge --inputs $INP --split navtrain --src-arm R8_LEGAL_s1 \
      --dst-arm R8_RT_s1 --retime-table $RT --retime-table-sha256 $RTSHA --retime-stage-file $RTSTAGE --tokens-file $SAMPLE \
      > $D/derive_rt_s1.log 2>&1
done
# ---- G4 (post): every seam complete, ZERO stand-ins (run_bridge8 hard-refuses missing frames only for split 'navtest'), KPR PASS ----
$PY - $OUT <<'EOF' || { echo ZZREFUSE_post_gate; exit 17; }
import glob, json, os, sys
bad = []
for mp in glob.glob(os.path.join(sys.argv[1], "ema*", "bridge", "seam_*.manifest.json")):
    m = json.load(open(mp))
    if m.get("partial") is not False or m.get("n_cv_standin_rows", 0) != 0 or m.get("n_missing", 1) != 0:
        bad.append(mp)
    if m.get("arm") in ("R8_LEGAL", "R8_LEGAL_s1") and (m.get("KPR") or {}).get("verdict") != "PASS":
        bad.append(mp + " KPR")
assert len(glob.glob(os.path.join(sys.argv[1], "ema*", "bridge", "seam_R8_RT*.manifest.json"))) == 4, "expected 4 R8_RT seams"
assert not bad, bad
EOF
$PY $T/rt_assert.py $FROZEN $RT $OUT/RT_ASSERT.json $OUT/ema40000/bridge $OUT/ema48852/bridge > $OUT/rt_assert.log 2>&1
grep -q '"verdict": "PASS"' $OUT/RT_ASSERT.json || { echo ZZREFUSE_rt_table; exit 18; }
echo ZZROLLSDONE
# ---- NEXT (CPU, NOT BUILT -- GROUNDWORK.md B5c-B5e): navtrain metric cache for the sample; score seams R8_RT + R8_RT_s1 at
#      both steps + STOP + CV + PRIOR_ha0p on the Thor CPU backend (navsim-cpu venv, <= 4 scorers, MemAvailable >= 35 GB,
#      NAVSIM_YIELD honoured); then the decision script (paired 40k - 48,852 PDMS per seed, log-cluster bootstrap B 2000,
#      floor |d_s0 - d_s1|, controls: token-shuffled pairing widens, 48,852 - 48,852 == 0 exactly, STOP and CV rows beside).
