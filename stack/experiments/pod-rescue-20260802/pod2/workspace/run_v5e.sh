#!/bin/bash
# v5e = v5d with the HELD-OUT GATE REMOVED (PI decision 2026-07-29).
#
# WHY: the in-loop gate fired at step 2000 and killed the run TWICE (C65, C66), burning ~7h
# on the same 1000->2000 stretch and never once producing a verdict. A run that repeatedly
# dies at the same step yields neither a model nor a gate result. The gate is now run
# OFFLINE against saved checkpoints instead, which cannot take the trainer down with it.
#
# Removing --heldout-gate also removes the whole C65/C66 failure mode by construction:
# heldout_gate._taniteval() is never imported, so its hard-derived sys.path insert
# (which ignores PYTHONPATH) can no longer force-load a stale taniteval tree.
#
# Everything else is byte-identical to run_v5d.sh so the run stays comparable.

OUT=/workspace/experiments/flagship-v5-w120-30k
mkdir -p "$OUT"
LOG=/workspace/v5e_run.log
ERR=/workspace/v5e_run.err
STATUS=/workspace/v5e_run.status

cd /workspace/TanitAD/stack || exit 90

echo "=== v5e (NO HELDOUT GATE) START $(date -u +%FT%TZ) ===" >> "$STATUS"

PYTHONPATH=/workspace/TanitAD/stack:/workspace/tev/taniteval OMP_NUM_THREADS=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -u scripts/train_flagship_v4.py   --v2-train-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl   --v2-val-cache   /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl   --v2-lru 8   --require-parity   --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical   --v2-subframe 176x624   --from-scratch   --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt   --out "$OUT"   --steps 30000 --batch 8 --accum 8 --lr-head 1e-4 --lr-trunk 1e-4   --warmup 2000 --workers 4 --eval-every 500 --save-every 1000 --rollout-k 4   --device cuda   >> "$LOG" 2>> "$ERR"

rc=$?
{
  echo "=== v5e EXIT $(date -u +%FT%TZ) rc=$rc ==="
  if [ $rc -gt 128 ]; then
    echo "    rc>128 => killed by signal $((rc-128))  (137=SIGKILL/OOM, 143=SIGTERM, 129=SIGHUP)"
  fi
  echo "    last stderr:"; tail -20 "$ERR" | sed 's/^/      /'
} >> "$STATUS"
