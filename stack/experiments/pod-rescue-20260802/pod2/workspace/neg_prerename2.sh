#!/bin/bash
# ⛔ NEGATIVE PROOF that the RENAME was load-bearing — CORRECTED.
#
# ⚠️ My first attempt used a SYMLINK under the old name. That is NOT a valid
# reproduction: `parity.corpus_key_of` calls `Path(path).resolve()`, which
# follows symlinks, so the old name resolved straight through to the new key and
# the run passed. Recorded because it looked exactly like an inert guard.
#
# The correct reproduction is a REAL directory under the old name holding
# HARDLINKS to the same payload bytes. `corpus_key_of` then sees a path with no
# registered key in it, which is the pre-rename state exactly.
set -u
export PYTHONPATH=/workspace/v5eval/stack
cd /workspace/v5eval/stack
TR=/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl
VA=/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
OLD=/workspace/v5eval/pai_wide120_v2png_train

rm -rf "$OLD"; mkdir -p "$OLD"
i=0
for f in "$VA"/*.v2ep.pt; do
  ln "$f" "$OLD/$(basename "$f")" 2>/dev/null || cp "$f" "$OLD/"
  i=$((i+1)); [ $i -ge 3 ] && break
done
cp "$VA/_geometry.json" "$OLD/" 2>/dev/null
echo "old-name dir: $(ls "$OLD" | grep -c 'v2ep.pt$') payloads (hardlinks; no bytes copied)"

echo
echo "### SYMLINK vs REAL DIR — why the first attempt was invalid"
ln -sfn "$TR" /tmp/symlinked_old_name
python3 - <<PY
import sys; sys.path.insert(0,"/workspace/v5eval/stack")
from tanitad.data import parity
for p in ("/tmp/symlinked_old_name", "$OLD", "$TR", "$VA"):
    print(f"  corpus_key_of({p}) = {parity.corpus_key_of(p)}")
PY
rm -f /tmp/symlinked_old_name

echo
echo "### TRAINER --require-parity against the OLD directory name"
python3 -u scripts/train_flagship_v4.py \
  --v2-train-cache "$OLD" --v2-val-cache "$VA" --require-parity \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 --from-scratch --out /tmp/x --print-launch 2>&1 \
  | grep -E "PREFLIGHT|PARITY VIOLATION|unregistered|register_v2_sibling|DIRECTORY NAME" | head -8
echo "EXIT: ${PIPESTATUS[0]}"

echo
echo "### EVALUATOR --require-parity against the OLD directory name"
python3 - <<PY
import sys
sys.path.insert(0, "/workspace/v5eval/stack")
sys.path.insert(0, "/workspace/v5eval/stack/scripts")
import argparse
from tanitad.geometry import add_geometry_args
from tanitad.data import parity
from tanitad.config import flagship4b_config
import eval_flagship_v4 as E
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
          "| names register_v2_sibling.py:", "register_v2_sibling" in t,
          "| names the DIRECTORY-NAME rule:", "DIRECTORY NAME contains" in t)
PY
rm -rf "$OLD"
