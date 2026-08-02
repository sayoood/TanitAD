#!/bin/bash
set -u
export PYTHONPATH=/workspace/v5eval/stack
cd /workspace/v5eval/stack
python3 /workspace/v5eval/merge_encoder_share.py \
  --inputs /workspace/v5eval/raw/encshare_p1_*.json /workspace/v5eval/raw/encshare_p2_*.json \
  --out /workspace/v5eval/raw/encoder_share_2026-07-27.json > /dev/null
python3 - <<'PY'
import json
d = json.load(open('/workspace/v5eval/raw/encoder_share_2026-07-27.json'))
print("micro_batch", d["micro_batch"], "passes", d["passes"])
for t, a in d["arms"].items():
    print(f"{t}: tokens={a['n_tokens']} imgs/step={a['images_encoded_per_step']} "
          f"encM={a['encoder_params_M']} totalM={a['total_params_M']}")
    print(f"   step  {a['full_step_s']['median']:.4f}s  spread {a['full_step_s']['spread_pct']}%  "
          f"samples {a['full_step_s']['samples']}")
    print(f"   enc   {a['encoder_only_s']['median']:.4f}s  spread {a['encoder_only_s']['spread_pct']}%  "
          f"samples {a['encoder_only_s']['samples']}")
    print(f"   step_ratio {a['step_ratio_vs_256x640']}  enc_ratio {a['encoder_ratio_vs_256x640']}  "
          f"share_standalone {a['encoder_share_of_step_standalone']}  "
          f"share_implied {a['encoder_share_implied_by_step_ratio']}")
print(json.dumps(d["headline"]["scope"]))
PY
echo "=== capacity ==="
python3 - <<'PY'
import glob, json
for p in sorted(glob.glob('/workspace/v5eval/raw/capacity_*.json')):
    d = json.load(open(p))
    for t, a in d["arms"].items():
        c = a["capacity"]
        print(t, "max_micro_batch", c["max_micro_batch"], "OOM_at", c["OOM_at"],
              "fits", c["fits"], "grad_ckpt", c["grad_checkpoint"],
              "gpu_total_MiB", c["gpu_total_MiB"])
PY
