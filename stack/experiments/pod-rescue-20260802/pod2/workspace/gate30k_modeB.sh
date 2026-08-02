#!/bin/bash
# The 30k gate MODE B, both goal modes, with taniteval ON THE PATH.
#
# WHY THE PATH MATTERS (the documented B4 trap, which just fired on my first run):
# eval_flagship_v4.py imports taniteval NON-FATALLY. Without it the run exits 0, writes a
# full-looking JSON, and silently drops the episode-cluster-bootstrap primary to null.
# The first oracle run "succeeded" exactly that way. Never read a v4 eval JSON without
# checking that driving_py_from_persisted_windows is non-null — this script checks it.
#
# The card requires BOTH goal modes reported as a pair (oracle = upper bound; produced =
# what a deployable stack could actually feed itself).

set -u
STACK=/workspace/TanitAD/stack
TEV=/workspace/tev/taniteval
OUT=/workspace/v4gate30k
CKPT=$OUT/v4fs_ckpt.pt
CFG=$OUT/v4fs_config.json
VAL=/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
ANCH=/workspace/experiments/flagship_v4_anchors_dense.pt

# wait out any eval still running so two forward passes never share the GPU
while pgrep -f eval_flagship_v4 > /dev/null 2>&1; do sleep 15; done

for MODE in oracle produced; do
  echo "=== MODE B / goal-mode=$MODE ==="
  cd "$STACK" || exit 1
  PYTHONPATH="$STACK:$TEV" OMP_NUM_THREADS=6 python3 -u scripts/eval_flagship_v4.py \
    --ckpt "$CKPT" \
    --head-config "$CFG" \
    --val-cache "$VAL" \
    --anchors-dense "$ANCH" \
    --goal-mode "$MODE" \
    --key "v4fs-30k-$MODE" \
    --out "$OUT/modeB_$MODE.json" \
    --results-dir "$OUT" \
    --device cuda > "$OUT/modeB_$MODE.log" 2>&1
  echo "  rc=$?"
  python3 - "$OUT/v4fs-30k-$MODE.json" << 'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print("  could not read result:", e); raise SystemExit
v = d.get("v4_diagnostics", {})
x = v.get("cross_check_ade_0_2s_selfcomputed_vs_driving_py", {})
dp = x.get("driving_py_from_persisted_windows")
print(f"  n_windows                = {v.get('n_windows')}")
print(f"  wp4_ade_0_2s_selfcomputed= {v.get('wp4_ade_0_2s_selfcomputed')}")
print(f"  wp4_oracle_ade_0_2s      = {v.get('wp4_oracle_ade_0_2s')}")
print(f"  dense_sel_gap            = {v.get('dense_headhorizons_sel_gap')}")
print(f"  dense_miss_at_2m         = {v.get('dense_headhorizons_miss_at_2m')}")
print(f"  seam_norm_ratio_max      = {v.get('seam_norm_ratio_max')}")
print(f"  driving_py (BOOTSTRAP)   = {dp}")
if dp is None:
    print("  PRIMARY STILL NULL - taniteval did not run; do NOT quote as a gate number.")
else:
    print(f"  primary computed; agree_within_1pct={x.get('agree_within_1pct')}")
PY
done
echo "=== GATE EVALS DONE ==="
