#!/bin/bash
# nt_k0_bridge.sh -- H-NT-DECIDE-1 control K0, the GPU half (PREREG_NT_DECIDE sec. 6): the fine-tune trainer's STEP-0
# saves (ckpt_0.pt RAW, ckpt_ema_0.pt EMA), read by the UNCHANGED bridge on the registered navtrain sample, must give
# R8_LEGAL rows BIT-IDENTICAL to the banked MOD2 theta_DEC rows (wmod2_DEC/bridge/seam_R8_LEGAL{,_s1}.npz); PDMS 76.36 /
# 75.68 then follows (the banked scores of those exact rows). The CPU half (nt_controls.py k0-weights: every tensor
# bit-equal to theta_DEC, md5 dad4cb5b...) PASSED on the 2026-10-21 Thor CPU smoke's saves.
#
# A COPY of r8ns_wisemod2_LAUNCH.sh's roll() -- same wrapper, same flags, same config ($R/ckpt/config.json: the refcv8
# launch config the banked rows were rolled with; the A1 checkpoints keep the architecture, so the bridge config is
# NOT the A1 run's), same post-gate -- with only the checkpoint swapped, plus a seam comparator. Thor CUDA, behind the
# GPU flock (blocking, -o), and ONLY once the GPU is free (it refuses otherwise):
#   nohup setsid bash nt_k0_bridge.sh <A1 run dir> > k0.out 2>&1 < /dev/null &    (it takes the GPU lock itself)
set -u
# ⛔ THOR GPU PROTOCOL (the Master Mind, 2026-10-21 ~23:00): every GPU step takes thor_gpu.lock AND finds the GPU empty;
# the GPU is never shared. The script ENFORCES it itself instead of trusting its caller: it re-execs under a BLOCKING
# flock (-o: the lock fd is not inherited by any child -- the supervisor-lock trap), then asserts the lock is really
# held (a non-blocking probe must FAIL), then refuses unless nvidia-smi lists no compute app.
GPU_LOCK=/home/nvidia/refcv7_post/thor_gpu.lock
NT_PENDING=/home/nvidia/r8ns/nt/NT_LAUNCH_PENDING   # MM 2026-10-22 00:45: the L7 chain yields to NT when it exists
if [ "${NT_K0_UNDER_LOCK:-}" != 1 ]; then
  touch "$NT_PENDING" || { echo "ZZREFUSE_cannot_write_yield_marker"; exit 16; }   # FIRST; left in place (the MM clears it)
  exec env NT_K0_UNDER_LOCK=1 flock -o "$GPU_LOCK" bash "$0" "$@"
fi
if flock -n "$GPU_LOCK" true 200>&-; then echo "ZZREFUSE_gpu_lock_not_held"; exit 15; fi
RUN=${1:?usage: nt_k0_bridge.sh <A1 run dir with ckpt_0.pt + ckpt_ema_0.pt>}
R=/home/nvidia/r8ns; C=$R/code; P=$C/pkg; T=$C/thor; N=$R/navtrain; OUT=$R/nt/k0_bridge
BANK_SEAMS=$R/navtrain_sel/wiseft/wmod2_DEC/bridge
PY=/home/nvidia/venvs/tanitad-train/bin/python
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy
export R7_GPU_LOCK=$OUT/r8ns_gpu_token_ntk0.lock
R7CODE=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/navsim/code
ROAD=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs/road_plane_navhard_warmup_logs.json
SAMPLE=$N/NAVTRAIN_SAMPLE_TOKENS.json; INP=$N/inputs/navtrain_sample_inputs.json.gz; BANK=$N/frame_bank; LOGS=$N/navsim_logs/trainval
mkdir -p $OUT; cd $OUT
[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo ZZREFUSE_gpu_busy; exit 14; }
for f in ckpt_0.pt ckpt_ema_0.pt; do [ -s $RUN/$f ] || { echo "ZZREFUSE_missing_$f"; exit 13; }; done
roll() {  # $1 = tag, $2 = ckpt, $3 = arms   (r8ns_wisemod2_LAUNCH.sh roll(), the checkpoint swapped)
  local TAG=$1 CK=$2 ARMS=$3 MD5 D TOK DEDUP
  MD5=$(md5sum $CK | cut -c1-32); D=$OUT/$TAG; mkdir -p $D
  R8_TEST_CKPT=$CK R8_TEST_CONFIG=$R/ckpt/config.json R8_TEST_DEVICE=cuda $PY $T/r8ns_cuda_controls.py $P $D/cuda_controls.json > $D/cuda_controls.log 2>&1
  [ "$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('K0_pass'))")" = True ] || { echo "ZZREFUSE_K0_$TAG"; return 1; }
  DEDUP=""; [ "$($PY -c "import json;print(json.load(open('$D/cuda_controls.json')).get('KD_pass'))")" = True ] && DEDUP="--exact-dedup"
  TOK=$($PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; print(gpu_lock.acquire('nt-k0',0,log=lambda m:None) or 'NONE')")
  [ "$TOK" != NONE ] || { echo "ZZREFUSE_token_$TAG"; return 1; }
  $PY $T/r8ns_bridge_wrap.py $P/code -- --split navtrain --arms $ARMS --inputs $INP --bank1 $BANK \
      --bank1-kind navtest --logs-root $LOGS --road-plane $ROAD --ckpt $CK --config $R/ckpt/config.json --ckpt-md5 $MD5 \
      --label "H-NT-DECIDE-1 K0, $TAG (the A1 trainer's step-0 save; must equal MOD2 theta_DEC bit for bit)" --device cuda \
      --precision auto --threads 6 --out $D/bridge --tokens-file $SAMPLE --gpu-lock-token $TOK $DEDUP > $D/bridge.log 2>&1
  $PY -c "import sys; sys.path.insert(0,'$R7CODE'); import gpu_lock; gpu_lock.release('$TOK')"
  for A in ${ARMS//,/ }; do
    $PY - $D/bridge $A <<'PYEOF' || { echo "ZZREFUSE_post_gate_${TAG}_$A"; return 1; }
import json, os, sys
m = json.load(open(os.path.join(sys.argv[1], f"seam_{sys.argv[2]}.manifest.json")))
assert m.get("partial") is False and m.get("n_cv_standin_rows", 0) == 0 and m.get("n_missing", 1) == 0 and m.get("n_rows") == 2089, m
PYEOF
    echo "ZZWROLLED_${TAG}_$A"
  done
}
roll k0_raw $RUN/ckpt_0.pt R8_LEGAL,R8_LEGAL_s1
roll k0_ema $RUN/ckpt_ema_0.pt R8_LEGAL,R8_LEGAL_s1
# the comparator: every array of every seam bit-identical to the banked theta_DEC seam (the artifact, not a score)
$PY - $OUT $BANK_SEAMS <<'PYEOF'
import json, os, sys
import numpy as np
out, bank = sys.argv[1], sys.argv[2]
res, ok = {}, True
for tag in ("k0_raw", "k0_ema"):
    for arm in ("R8_LEGAL", "R8_LEGAL_s1"):
        a = os.path.join(out, tag, "bridge", f"seam_{arm}.npz"); b = os.path.join(bank, f"seam_{arm}.npz")
        if not (os.path.isfile(a) and os.path.isfile(b)):
            res[f"{tag}/{arm}"] = {"status": "MISSING", "a": os.path.isfile(a), "b": os.path.isfile(b)}; ok = False; continue
        za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
        keys = sorted(set(za.files) | set(zb.files))
        neq = [k for k in keys if k not in za.files or k not in zb.files or za[k].shape != zb[k].shape
               or za[k].dtype != zb[k].dtype or not np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f")]
        res[f"{tag}/{arm}"] = {"n_arrays": len(keys), "unequal": neq[:20], "bit_identical": not neq}
        ok = ok and not neq
json.dump({"control": "K0-bridge", "PASS": ok, "seams": res, "bank": bank}, open(os.path.join(out, "K0_BRIDGE.json"), "w"), indent=1)
print("ZZK0_" + ("PASS" if ok else "FAIL"))
PYEOF
echo ZZK0_DONE
