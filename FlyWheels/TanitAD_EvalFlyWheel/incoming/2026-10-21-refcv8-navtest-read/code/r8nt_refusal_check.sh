#!/bin/bash
# r8nt_refusal_check.sh -- G1's mutation arm: does run_bridge8 HARD-REFUSE a navtest token whose frames are missing (instead of writing a
# declared CV stand-in)?  A copy of the bank index with ONE token removed (frames symlinked, nothing else changed) + that token alone.
# Expected: the bridge exits with "... is not in the 416 bank -- refusing" and writes NO model row and NO stand-in row.  CPU only.
# Marker: QQREFUSAL <json>.
set -u
R=/home/nvidia/r8ns; C=$R/code; P=$C/pkg; X=$R/navtest_read; NT=$R/navtest; D=$X/refusal_check
PY=/home/nvidia/venvs/tanitad-train/bin/python
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt CUDA_VISIBLE_DEVICES=-1
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy
R6IN=$C/tree/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim/raw/inputs
rm -rf $D; mkdir -p $D/bank_minus1
for f in $NT/frame_bank/frames_s*.npy; do ln -s $f $D/bank_minus1/; done
$PY - $NT/frame_bank/index.json $D/bank_minus1/index.json $X/prep/smoke20_tokens.json $D/one_token.json <<'EOF'
import json, sys
idx = json.load(open(sys.argv[1])); sm = json.load(open(sys.argv[3]))
t = sm["tokens"][0]
del idx["tokens"][t]
json.dump(idx, open(sys.argv[2], "w"))
json.dump({"tokens": [t], "token_log": {t: sm["token_log"][t]}}, open(sys.argv[4], "w"))
EOF
$PY $P/code/run_bridge8.py --split navtest --arms R8_LEGAL --inputs $NT/inputs/navtest_inputs.json.gz --bank1 $D/bank_minus1 \
    --bank1-kind navtest --logs-root $NT/nsdata/openscene/navsim_logs/test --road-plane $R6IN/road_plane_navhard_warmup_logs.json \
    --ckpt $R/ckpt/ckpt_ema_48852.pt --config $R/ckpt/config.json --device cpu --threads 6 --out $D/out \
    --tokens-file $D/one_token.json > $D/bridge.log 2>&1
$PY - $D <<'EOF'
import json, os, sys
d = sys.argv[1]
log = open(os.path.join(d, "bridge.log"), encoding="utf-8", errors="replace").read()
rp = os.path.join(d, "out", "rows_R8_LEGAL.jsonl")
rows = [l for l in open(rp)] if os.path.exists(rp) else []
rec = {"refused_message_seen": "is not in the 416 bank -- refusing" in log, "rows_written": len(rows),
       "seam_written": os.path.exists(os.path.join(d, "out", "seam_R8_LEGAL.npz"))}
rec["verdict"] = "PASS" if rec["refused_message_seen"] and rec["rows_written"] == 0 and not rec["seam_written"] else "FAIL"
json.dump(rec, open(os.path.join(d, "REFUSAL_CHECK.json"), "w"), indent=1)
print("QQREFUSAL " + json.dumps(rec))
EOF
