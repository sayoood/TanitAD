#!/bin/bash
# ⛔ NEGATIVE PROOF that the rename + registration were load-bearing.
# A symlink reproduces the PRE-RENAME state exactly: the SAME bytes under the
# old directory name, whose corpus key `corpus_key_of` cannot resolve.
set -u
export PYTHONPATH=/workspace/v5eval/stack
cd /workspace/v5eval/stack
TR=/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl
VA=/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
OLD=/tmp/pai_wide120_v2png_train
rm -f "$OLD"; ln -s "$TR" "$OLD"

echo "### corpus_key_of on the OLD name vs the NEW name"
python3 - <<PY
import sys; sys.path.insert(0,"/workspace/v5eval/stack")
from tanitad.data import parity
for p in ("$OLD", "$TR", "$VA"):
    print(f"  {p} -> {parity.corpus_key_of(p)}")
PY

echo
echo "### --require-parity against the OLD directory name (expect REFUSAL)"
python3 -u scripts/train_flagship_v4.py \
  --v2-train-cache "$OLD" --v2-val-cache "$VA" --require-parity \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 --from-scratch --out /tmp/x --print-launch 2>&1 | tail -22
echo "EXIT: ${PIPESTATUS[0]}"

echo
echo "### the EVALUATOR under --require-parity against the OLD name (expect REFUSAL)"
python3 - <<PY
import sys
sys.path.insert(0, "/workspace/v5eval/stack")
sys.path.insert(0, "/workspace/v5eval/stack/scripts")
import argparse
from tanitad.geometry import add_geometry_args
from tanitad.data import parity
import eval_flagship_v4 as E
from tanitad.config import flagship4b_config
p = argparse.ArgumentParser()
p.add_argument("--v2-val-cache", nargs="+"); p.add_argument("--v2-lru", type=int, default=64)
p.add_argument("--v2-subframe"); p.add_argument("--require-parity", action="store_true")
add_geometry_args(p)
a = p.parse_args(["--v2-val-cache", "$OLD", "--v2-subframe", "176x624",
                  "--require-parity", "--frame-h", "256", "--frame-w", "640",
                  "--frame-hfov", "120", "--projection", "cylindrical"])
cf, mf = E.resolve_eval_frames(a, flagship4b_config())
try:
    E.build_v2_val_episodes(a, cache_frame=cf, train_frame=mf, verbose=False)
    print("  NOT REFUSED  <-- GUARD IS INERT")
except parity.ParityViolation as ex:
    t = str(ex)
    print("  REFUSED:", "unregistered v2 cache" in t,
          "| names register_v2_sibling:", "register_v2_sibling" in t,
          "| names the DIRECTORY NAME rule:", "DIRECTORY NAME contains" in t)
PY
rm -f "$OLD"
