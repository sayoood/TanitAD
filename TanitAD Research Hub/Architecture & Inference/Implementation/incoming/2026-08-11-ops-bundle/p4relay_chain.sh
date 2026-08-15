#!/bin/bash
# Pod4: pull staging from HF, then stack-sync + P1 lead-gap rerun (class filter).
python3 /workspace/p4_pull.py >> /tmp/p4relay.log 2>&1
if ! grep -q RELAY_DOWN_DONE /tmp/p4relay.log; then
  echo "P4RELAY_EXIT=PULL_FAILED" >> /tmp/p4relay.log
  exit 1
fi
cd /workspace/TanitAD || { echo "P4RELAY_EXIT=NO_CHECKOUT" >> /tmp/p4relay.log; exit 1; }
git fetch origin claude/tanitad-resumption-handoff-92zx39 >> /tmp/p4relay.log 2>&1
git checkout -B claude/tanitad-resumption-handoff-92zx39 origin/claude/tanitad-resumption-handoff-92zx39 >> /tmp/p4relay.log 2>&1
if ! grep -q "lookup_classes" stack/scripts/probe_latent_state.py; then
  echo "P4RELAY_EXIT=SYNC_FAILED_FILTER_MISSING" >> /tmp/p4relay.log
  exit 1
fi
echo "STACK_SYNC_OK $(git rev-parse --short HEAD)" >> /tmp/p4relay.log
cd /workspace/TanitAD/stack
export PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6
mkdir -p /workspace/experiments/p1-rerun-clsfilter
python3 -u scripts/probe_latent_state.py \
  --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 \
  --join-file /workspace/data/p8_join/combined140.jsonl \
  --out /workspace/experiments/p1-rerun-clsfilter > /tmp/p1_rerun.log 2>&1
echo "P1RERUN_EXIT=$?" >> /tmp/p1_rerun.log
echo "P4RELAY_EXIT=done" >> /tmp/p4relay.log
