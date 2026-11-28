#!/bin/bash
# AMENDMENT 1(b) control: derive_rt_seed.py on the SEED-0 rows must reproduce run_bridge8 --retime's R8_RT exactly (per checkpoint).
set -u
R=/home/nvidia/r8ns; C=$R/code; OUT=$R/navtrain_sel; N=$R/navtrain
PY=/home/nvidia/venvs/tanitad-train/bin/python
unset REFCV6_KIT REFCV6_REMAP_OVERRIDES REFCV6_REPO REFCV7_KIT REFCV7_REMAP_OVERRIDES REFCV7_REPO R8_MUTATION R7_MUTATION
export TANITAD_REPO=$C/tree R8_LAUNCH_TREE=$C/tree R8_EVAL_KIT=/home/nvidia R8_BATTERY_CODE=$C/batt PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 PYTHONPATH=$C/tree/stack:$C/tree/taniteval:$R/overlay_scipy
RT=$C/retime/refcv8_retime_deploy_gains_pf.json; RTSHA=3319b003fecc83577430f07dad762bce0a9a0b8594e9748e9a61bc4d264136b7
for ck in ema40000 ema48852; do
  D=$OUT/ctrl_rt_seed0/$ck; mkdir -p $D
  cp $OUT/$ck/bridge/rows_R8_LEGAL.jsonl $D/
  $PY $C/thor/derive_rt_seed.py --pkg-code $C/pkg/code --out $D --inputs $N/inputs/navtrain_sample_inputs.json.gz --split navtrain \
      --src-arm R8_LEGAL --dst-arm R8_RT --retime-table $RT --retime-table-sha256 $RTSHA --retime-stage-file $C/retime/retime_stage.py \
      --tokens-file $N/NAVTRAIN_SAMPLE_TOKENS.json > $D/derive.log 2>&1
  $PY - $OUT/$ck/bridge $D <<'PY'
import json, sys, numpy as np, os
a_dir, b_dir = sys.argv[1], sys.argv[2]
def rows(p):
    out = {}
    if os.path.exists(p):
        for ln in open(p):
            if ln.strip():
                r = json.loads(ln); out[r["token"]] = r
    return out
ra, rb = rows(f"{a_dir}/rows_R8_RT.jsonl"), rows(f"{b_dir}/rows_R8_RT.jsonl")
res = {"n_bridge_rows": len(ra), "n_ctrl_rows": len(rb)}
if ra and rb:
    same_keys = set(ra) == set(rb)
    diff = [t for t in ra if t in rb and (ra[t]["knots"] != rb[t]["knots"] or ra[t]["poses"] != rb[t]["poses"])]
    res.update(rows_same_tokens=same_keys, rows_differing=len(diff))
za, zb = np.load(f"{a_dir}/seam_R8_RT.npz", allow_pickle=False), np.load(f"{b_dir}/seam_R8_RT.npz", allow_pickle=False)
res["npz_keys_a"] = sorted(za.files); res["npz_keys_b"] = sorted(zb.files)
eq = {}
for k in za.files:
    if k in zb.files:
        x, y = za[k], zb[k]
        eq[k] = bool(x.shape == y.shape and (np.array_equal(x, y) if x.dtype.kind not in "fc" else np.array_equal(x, y, equal_nan=True)))
res["npz_equal_per_key"] = eq
res["verdict"] = "PASS" if (eq and all(eq.values()) and set(za.files) == set(zb.files) and res.get("rows_differing", 0) == 0
                            and res.get("rows_same_tokens", True)) else "FAIL"
json.dump(res, open(f"{b_dir}/CTRL_RESULT.json", "w"), indent=1)
print("ZZCTRL " + json.dumps(res))
PY
done
