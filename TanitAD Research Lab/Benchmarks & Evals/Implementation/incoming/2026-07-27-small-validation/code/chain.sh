#!/bin/bash
# TanitAD small validation — matched short runs, geometry contrast.
# RE-SCOPED 2026-07-27 after a MEASURED cost finding: at --batch 8 --accum 8 one
# OPTIMIZER step is 8 micro-batches, so a step costs 8.58 s (256x256) /
# 13.56 s (256x640) / 10.39 s (176x624) -- 6000 steps x 3 arms = 54 GPU-h.
# 1500 steps keeps the SAME matched design at 13.5 GPU-h.
# Arms run SEQUENTIALLY (one A40). Order = PRIORITY order: the PI's contrast
# (A vs B) completes first; C is the rig-fix separator and may be cut.
cd /workspace/TanitAD/stack
export PYTHONPATH=/workspace/TanitAD/stack
export OMP_NUM_THREADS=6

RAW_TR=/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894
RAW_VA=/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11
V2_TR=/workspace/data/physicalai-train-e438721ae894-w120-256x640cyl
V2_VA=/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl
ANCH=/workspace/experiments/flagship_v4_anchors_dense.pt

COMMON="--from-scratch --anchors-dense $ANCH --steps 1500 --batch 8 --accum 8 \
 --lr-head 1e-4 --lr-trunk 1e-4 --warmup 100 --workers 8 \
 --phase-a-steps 100 --phase-b-steps 400 --gate-step 400 \
 --eval-every 500 --save-every 500 --eval-episodes 40 --rollout-k 4 \
 --heldout-gate --heldout-every 500 --heldout-episodes 8 --heldout-patience 2 \
 --heldout-stride 8 --heldout-nboot 2000 --heldout-goal dropped \
 --seed 0 --device cuda"

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "[chain] START $(stamp)  chain pid=$$"

echo "[chain] === ARM A_OLD  (raw epcache 256x256 @51.4deg) $(stamp) ==="
python3 -u scripts/train_flagship_v4.py \
  --train-cache $RAW_TR --val-cache $RAW_VA \
  --out /workspace/smallval/A_old-256x256 \
  $COMMON > /tmp/smallval_A_old.log 2>&1
echo "[chain] ARM A rc=$? $(stamp)"
touch /workspace/smallval/A_DONE

echo "[chain] === ARM B_WIDE (v2 256x640 @120deg cyl, subframe none) $(stamp) ==="
python3 -u scripts/train_flagship_v4.py \
  --v2-train-cache $V2_TR --v2-val-cache $V2_VA --v2-lru 64 --require-parity \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe none \
  --out /workspace/smallval/B_wide-256x640 \
  $COMMON > /tmp/smallval_B_wide.log 2>&1
echo "[chain] ARM B rc=$? $(stamp)"
touch /workspace/smallval/B_DONE

echo "[chain] === ARM C_V5 (v2 176x624 rig-clean) $(stamp) ==="
python3 -u scripts/train_flagship_v4.py \
  --v2-train-cache $V2_TR --v2-val-cache $V2_VA --v2-lru 64 --require-parity \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --out /workspace/smallval/C_v5-176x624 \
  $COMMON > /tmp/smallval_C_v5.log 2>&1
echo "[chain] ARM C rc=$? $(stamp)"
touch /workspace/smallval/C_DONE

echo "[chain] DONE $(stamp)"
touch /workspace/smallval/CHAIN_DONE
