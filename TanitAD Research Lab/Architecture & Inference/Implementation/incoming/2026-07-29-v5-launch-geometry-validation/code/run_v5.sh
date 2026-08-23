#!/bin/bash
# v5 176x624 launcher WITH an exit-code trap.
#
# WHY: the first launch died silently — no traceback, no OOM line, log simply stopped
# after the step-0 canary row, process gone. A trainer that vanishes without a reason is
# unfixable, so this wrapper records the exit status and the signal, and keeps stderr
# separate from stdout so a crash cannot be swallowed by the JSON log stream.
#
# setsid + </dev/null fully detaches into a NEW SESSION so no SIGHUP from a closing ssh
# channel can reach it (the documented "use ssh -f, not cmd &" trap in this program).

OUT=/workspace/experiments/flagship-v5-w120-30k
mkdir -p "$OUT"
LOG=/workspace/v5c_run.log
ERR=/workspace/v5c_run.err
STATUS=/workspace/v5c_run.status

cd /workspace/TanitAD/stack || exit 90

echo "=== v5 START $(date -u +%FT%TZ) ===" >> "$STATUS"

PYTHONPATH=/workspace/TanitAD/stack \
OMP_NUM_THREADS=6 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u scripts/train_flagship_v4.py \
  --v2-train-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache   /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --v2-lru 8 \
  --require-parity \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --from-scratch \
  --anchors-dense /workspace/experiments/flagship_v4_anchors_dense.pt \
  --out "$OUT" \
  --steps 30000 --batch 8 --accum 8 --lr-head 1e-4 --lr-trunk 1e-4 \
  --warmup 2000 --workers 4 --eval-every 500 --save-every 1000 --rollout-k 4 \
  --heldout-gate --heldout-every 2000 --heldout-episodes 8 --heldout-patience 2 \
  --device cuda \
  >> "$LOG" 2>> "$ERR"

rc=$?
{
  echo "=== v5 EXIT $(date -u +%FT%TZ) rc=$rc ==="
  if [ $rc -gt 128 ]; then
    echo "    rc>128 => killed by signal $((rc-128))  (137=SIGKILL/OOM, 143=SIGTERM, 129=SIGHUP)"
  fi
  echo "    last stderr:"; tail -20 "$ERR" | sed 's/^/      /'
} >> "$STATUS"

# (appended) memory sampler ran alongside — pod2 is cgroup-limited to ~50 GB and `free`
# shows the HOST's 503 GB, so the limit is invisible to the usual tool. Same class as
# "never judge pod disk with df".
