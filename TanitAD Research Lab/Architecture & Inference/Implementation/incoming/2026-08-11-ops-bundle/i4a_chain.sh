#!/bin/bash
# I4a (WM_PHYSICS_PROOF): imagination ablation attribution triplet on v5f-30k.
# Waits for the W4r->W7 chain to release the GPU, syncs stack, verifies the
# flag exists in the RUNNING checkout (runbook: a pod launch from a stale
# checkout resurrects fixed bugs), then runs intact/zero/shuffle on the 881 grid.
until grep -q W7W4R_EXIT /tmp/w7_w4r.log 2>/dev/null; do sleep 300; done
cd /workspace/TanitAD || exit 1
git fetch origin claude/tanitad-resumption-handoff-92zx39 >> /tmp/i4a.log 2>&1
git checkout -B claude/tanitad-resumption-handoff-92zx39 origin/claude/tanitad-resumption-handoff-92zx39 >> /tmp/i4a.log 2>&1
if ! grep -q "imagination-ablate" stack/scripts/eval_flagship_v4.py; then
  echo I4A_EXIT=SYNC_FAILED_FLAG_MISSING >> /tmp/i4a.log
  exit 1
fi
cd /workspace/TanitAD/stack
export PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6
mkdir -p /workspace/experiments/i4a
for MODE in none zero shuffle; do
  python3 -u scripts/eval_flagship_v4.py \
    --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
    --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
    --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
    --v2-subframe 176x624 --key flagship-v5f-w120-30k-i4a-$MODE \
    --cond-imagination auto --imagination-ablate $MODE \
    --out /workspace/experiments/i4a/i4a_$MODE.json >> /tmp/i4a.log 2>&1
  echo I4A_${MODE}_EXIT=$? >> /tmp/i4a.log
done
echo I4A_EXIT=done >> /tmp/i4a.log
