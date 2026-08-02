#!/bin/bash
# ⭐ THE EXPERIMENT THE PI ACTUALLY ASKED FOR: v1's ARCHITECTURE on the BIGGER CORPUS.
#
# WHY THIS EXISTS: flagship-v2corpus-30k was launched with BOTH --v2-cache AND --v2.
#   --v2-cache = a DATA LOADER flag. Its own help: "windows stay contract-identical
#                (same keys/shapes/dtypes/labels)". It only changes WHICH corpus is read.
#   --v2       = a TEN-LEVER ARCHITECTURE PACK (ego->planners, ego-dropout .25, fa-dropout .3,
#                goal-decode, nav-dropout .5, gated-intent, anchor-tactical, speed-input,
#                invdyn-gradscale .25, and rollout_k 12 instead of 4).
# The PI wanted ONLY the corpus change. This run is v1's flags + the v2bal corpus: NO --v2.
#
# Contrast this gives us, which v2corpus could NOT:
#   this arm vs v1        -> the CORPUS effect, architecture held fixed  (the PI's question)
#   this arm vs v2corpus  -> the LEVER-PACK effect, corpus held fixed
set -u
OUT=/workspace/experiments/flagship-v1arch-v2bal-30k
LOG=/workspace/v1v2bal_run.log
ERR=/workspace/v1v2bal_run.err
ST=/workspace/v1v2bal_run.status
mkdir -p "$OUT"
echo "=== v1arch-on-v2bal START $(date -u +%FT%TZ) ===" >> "$ST"
cd /workspace/TanitAD/stack || exit 90
export PYTHONPATH=/workspace/TanitAD/stack
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=6
python3 -u scripts/train_flagship4b.py   --v2-cache /workspace/rescue/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d   --config flagship4b --sigreg-free-dims 64   --rollout-k 4 --speed-input   --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint   --lr 3e-4 --warmup 2000 --workers 8 --v2-lru 64 --guard-limit-gb 45   --ckpt-every 1000 --log-every 50   --out "$OUT" >> "$LOG" 2>> "$ERR"
rc=$?
{ echo "=== v1arch-on-v2bal EXIT $(date -u +%FT%TZ) rc=$rc ==="
  [ $rc -gt 128 ] && echo "   killed by signal $((rc-128)) (137=SIGKILL/OOM)"
  echo "   last stderr:"; tail -20 "$ERR" | sed 's/^/     /'; } >> "$ST"
