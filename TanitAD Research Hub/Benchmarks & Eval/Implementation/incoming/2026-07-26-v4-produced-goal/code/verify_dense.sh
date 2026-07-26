#!/usr/bin/env bash
# Verify the ADDITIVE dense-path emission:
#   (a) `pred` is STILL bit-identical to the pre-existing baseline
#   (b) pred_dense/gt_dense are present and self-consistent with pred/gt
#   (c) taniteval.lateral.block now RUNS natively (10 Hz) on a v4 windows dump
export PYTHONPATH=/root/v4eval/stack:/root/taniteval:/root/v4eval/stack/scripts
export TANITEVAL_STACK_OVERRIDE=/root/v4eval/stack
M=/workspace/models/flagship-v4-fromscratch-15k
OUT=/root/v4eval/results_goalmode
cd /root/v4eval/stack/scripts || exit 1

python3 -u eval_flagship_v4.py \
  --ckpt "$M"/ckpt_step15000.pt --anchors-dense "$M"/flagship_v4_anchors_dense.pt \
  --head-config "$M"/config.json \
  --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \
  --goal-mode oracle --key v4-15k-oracle-dense \
  --out "$OUT/v4-15k-oracle-dense.json" --results-dir "$OUT" \
  --episodes 40 --stride 8 --batch 16 --device cuda --skip-bench --skip-driving \
  2>&1 | tail -4
echo "EXIT=$?"

python3 - <<'PY'
import torch, hashlib, numpy as np, json, sys
sys.path.insert(0, "/root/taniteval")
from taniteval import lateral as L
new  = torch.load("/root/v4eval/results_goalmode/windows_v4-15k-oracle-dense.pt",
                  map_location="cpu", weights_only=False)
base = torch.load("/root/v4eval/results/windows_flagship-v4-fromscratch-15k.pt",
                  map_location="cpu", weights_only=False)
md5 = lambda t: hashlib.md5(np.ascontiguousarray(t.numpy()).tobytes()).hexdigest()
print("keys:", sorted(new.keys()))
print("(a) pred bit-identical to baseline:", torch.equal(new["pred"], base["pred"]),
      "| md5", md5(new["pred"])[:12], "==", md5(base["pred"])[:12])
pd, gd = new["pred_dense"], new["gt_dense"]
print("(b) pred_dense", tuple(pd.shape), "gt_dense", tuple(gd.shape),
      "dense_steps", new["dense_steps"][:3], "...", new["dense_steps"][-1],
      "dt_s", new["dt_s"])
idx = [k - 1 for k in new["wp_steps"]]
print("    pred == pred_dense[:, wp_steps-1]:",
      torch.equal(new["pred"], pd[:, idx]))
print("    gt vs gt_dense[:, wp_steps-1] max|d| (different code paths):",
      round(float((new["gt"] - gd[:, idx]).abs().max()), 6))
b = L.block(new, n_boot=500)
print("(c) lateral.block skipped:", b.get("skipped", False))
print("    surface_dt_s:", b.get("dt_s"), "horizon_K:", b.get("horizon_K"),
      "horizon_s:", b.get("horizon_s"))
print("    energy_share_longitudinal:",
      b["energy_share"]["longitudinal_share_of_squared_error"])
print("    growth cross x", b["growth"]["cross_growth_final"],
      "vs along x", b["growth"]["along_growth_final"],
      "| cross faster by x", b["growth"]["cross_grows_faster_by"])
print("    axis_check verified:", b.get("axis_check", {}).get("verified"))
print("    VERDICT:", b.get("verdict"))
json.dump({"dense_verification": {
    "pred_bit_identical_to_baseline": bool(torch.equal(new["pred"], base["pred"])),
    "pred_md5": md5(new["pred"]), "baseline_pred_md5": md5(base["pred"]),
    "pred_dense_shape": list(pd.shape), "gt_dense_shape": list(gd.shape),
    "dense_steps": new["dense_steps"], "dt_s": new["dt_s"],
    "pred_equals_dense_subselection": bool(torch.equal(new["pred"], pd[:, idx])),
    "gt_vs_gt_dense_max_abs_diff": float((new["gt"] - gd[:, idx]).abs().max()),
    "lateral_block_runs": not bool(b.get("skipped")),
    "lateral_block": {k: b[k] for k in
                      ("dt_s", "horizon_K", "horizon_s", "energy_share",
                       "growth", "axis_check", "verdict") if k in b},
}}, open("/root/v4eval/results_goalmode/DENSE_PATH_VERIFICATION.json", "w"),
    indent=2, default=str)
print("-> /root/v4eval/results_goalmode/DENSE_PATH_VERIFICATION.json")
PY
