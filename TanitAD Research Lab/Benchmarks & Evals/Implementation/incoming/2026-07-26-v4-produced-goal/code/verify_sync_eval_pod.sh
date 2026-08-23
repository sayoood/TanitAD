#!/usr/bin/env bash
# POST-SYNC VERIFICATION on the eval pod.
#  (1) import smoke — the synced trees import at all
#  (2) MODE A on flagship v1 -> must reproduce 0.4271 (registry full-set)
#  (3) taniteval.lateral.block must RUN (needs the dense rollout path)
export PYTHONPATH=/root/v4eval/stack:/root/taniteval:/root/v4eval/stack/scripts
cd /root/v4eval/stack/scripts || exit 1

echo "=== (1) IMPORT SMOKE ==="
python3 - <<'PY'
import importlib, sys
mods = ["taniteval.rollout", "taniteval.lateral", "taniteval.ci",
        "taniteval.driving", "taniteval.corridor", "taniteval.hierarchy_guard",
        "taniteval.refc_eval", "tanitad.data.parity", "goal_provenance",
        "eval_flagship_v4", "train_flagship_v4", "flagship_v4_data"]
bad = []
for m in mods:
    try:
        importlib.import_module(m); print(f"  ok   {m}")
    except Exception as e:
        bad.append(m); print(f"  FAIL {m}: {type(e).__name__}: {e}")
import taniteval.rollout as R
print("  rollout has dense_speed_profile:", hasattr(R, "dense_speed_profile"))
import inspect
print("  rollout.collect emits pred_dense:",
      "pred_dense" in inspect.getsource(R.collect))
sys.exit(1 if bad else 0)
PY
echo "IMPORT_EXIT=$?"

echo "=== (2) MODE A on flagship v1 (target 0.4271, prior run 0.42148) ==="
python3 -u eval_flagship_v4.py \
  --ckpt /root/models/flagship-30k/ckpt.pt --canary-only \
  --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \
  --key v1-postsync --out /root/v4eval/results/v1-postsync.json \
  --episodes 40 --stride 8 --batch 16 --device cuda 2>&1 | tail -40
echo "MODEA_EXIT=$?"
