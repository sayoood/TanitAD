#!/bin/bash
# ⛔ DEMONSTRATE THE PREFLIGHT FIX FAILING — on the REAL pod2 cache, both directions.
#
# The defect (V5_EVALUABLE.md §9.2): `--print-launch --require-parity` printed
# `PREFLIGHT: OK` against a v2 cache whose corpus key resolves to None.
#
# ⚠️ The negative control uses HARDLINKS under a real directory of the old name,
# NOT a symlink: `parity.corpus_key_of` calls `Path.resolve()`, so a symlinked
# old name reads THROUGH to the renamed target and the guard would read as inert
# on a test that was itself invalid. (A sibling stream made exactly that mistake
# and self-corrected it — V5_EVALUABLE.md §5.1.)
#
# ⛔ Hardlinks share inodes: this creates NO copy of the ~80 GB payloads and
# deletes only the link directory afterwards.
set -u
STACK=${STACK:-/workspace/v5gate/stack}
GOOD=/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl
VAL=/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
BAD=/workspace/v5gate/prerename/pai_wide120_v2png_train
export PYTHONPATH=$STACK
cd "$STACK"

common() {
  echo "  --v2-train-cache $1"
  python3 -u scripts/train_flagship_v4.py \
    --v2-train-cache "$1" --v2-val-cache "$VAL" \
    --v2-lru 64 --require-parity \
    --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
    --v2-subframe 176x624 --from-scratch \
    --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
    --out /workspace/v5gate/run/DEMO \
    --steps 30000 --batch 8 --accum 8 --warmup 2000 --workers 8 \
    --eval-every 500 --save-every 1000 --rollout-k 4 \
    --heldout-gate --heldout-every 2000 --heldout-episodes 8 \
    --heldout-patience 2 --device cuda --print-launch 2>&1 \
    | grep -E 'PREFLIGHT|PARITY|register_v2_sibling|DIRECTORY NAME|v2 VERIFIED|COULD NOT'
  echo "  EXIT=${PIPESTATUS[0]}"
}

echo "=========== 0. corpus_key_of, both spellings (the ground truth) ==========="
rm -rf /workspace/v5gate/prerename
mkdir -p "$BAD"
# hardlink every payload into a REAL directory carrying the pre-rename name
n=0
for f in "$GOOD"/*.v2ep.pt; do ln "$f" "$BAD/$(basename "$f")"; n=$((n+1)); done
echo "  hardlinked $n payloads into $BAD (no copy: same inodes)"
python3 - <<PY
import sys
sys.path.insert(0, "$STACK")
from tanitad.data import parity
for p in ("$GOOD", "$BAD"):
    print(f"  corpus_key_of({p}) = {parity.corpus_key_of(p)}")
PY

echo
echo "=========== 1. ⛔ RED — the pre-rename directory (the measured defect) ====="
common "$BAD"

echo
echo "=========== 2. ⭐ GREEN — the renamed, registered directory ==============="
common "$GOOD"

echo
echo "=========== 3. the THIRD state — a cache not on this host ================"
STACK=$STACK python3 -u scripts/train_flagship_v4.py \
  --v2-train-cache /workspace/data/not-here-at-all \
  --v2-val-cache /workspace/data/nor-this \
  --require-parity --frame-h 256 --frame-w 640 --frame-hfov 120 \
  --projection cylindrical --v2-subframe 176x624 --from-scratch \
  --out /workspace/v5gate/run/DEMO --steps 30000 --batch 8 --accum 8 \
  --warmup 2000 --workers 8 --eval-every 500 --save-every 1000 --rollout-k 4 \
  --heldout-gate --heldout-every 2000 --heldout-episodes 8 \
  --heldout-patience 2 --device cuda --print-launch 2>&1 \
  | grep -E 'PREFLIGHT|COULD NOT BE CHECKED'
echo "  EXIT=${PIPESTATUS[0]}"

echo
echo "=========== cleanup: remove ONLY the hardlink directory =================="
rm -rf /workspace/v5gate/prerename
ls -d /workspace/v5gate/prerename 2>/dev/null || echo "  removed"
echo "  payloads intact in the real cache: $(ls $GOOD/*.v2ep.pt | wc -l)"
