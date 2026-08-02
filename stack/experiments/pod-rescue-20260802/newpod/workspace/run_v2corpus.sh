#!/bin/bash
# FINISH the 50h v2-corpus arm: 24,550 -> 30,000. PI directive 2026-07-29.
# Command copied VERBATIM from the original run manifest (pod1 /proc/699286/cmdline),
# with only the two PATHS repointed at this host's copies. Nothing else differs, so the
# resumed segment stays comparable with the 24,550 steps already trained.
#   corpus  physicalai-v2bal-4b7eeeac222d  (9,000 clips / 49.742 h)
# The trainer AUTO-RESUMES from ckpt.pt in --out (prints "[resume] ... resuming at step N").
set -u
OUT=/workspace/experiments/flagship-v2corpus-30k
LOG=/workspace/v2corpus_run.log
ERR=/workspace/v2corpus_run.err
ST=/workspace/v2corpus_run.status
mkdir -p "$OUT"
# stage the rescued checkpoint so the trainer resumes rather than restarting from 0
if [ ! -f "$OUT/ckpt.pt" ]; then
  cp /workspace/rescue/experiments/flagship-v2corpus-30k/ckpt.pt "$OUT/ckpt.pt" || exit 91
  cp /workspace/rescue/experiments/flagship-v2corpus-30k/config.json "$OUT/" 2>/dev/null
fi
echo "=== v2corpus START $(date -u +%FT%TZ) ckpt=$(stat -c %s "$OUT/ckpt.pt") ===" >> "$ST"
cd /workspace/TanitAD/stack || exit 90
export PYTHONPATH=/workspace/TanitAD/stack
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=6
python3 -u scripts/train_flagship4b.py \
  --v2-cache /workspace/rescue/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d \
  --config flagship4b --v2 --sigreg-free-dims 64 \
  --steps 30000 --batch-size 16 --accum 4 --grad-checkpoint \
  --lr 3e-4 --warmup 2000 --workers 8 --v2-lru 64 --guard-limit-gb 45 \
  --ckpt-every 1000 --log-every 50 \
  --out "$OUT" >> "$LOG" 2>> "$ERR"
rc=$?
{ echo "=== v2corpus EXIT $(date -u +%FT%TZ) rc=$rc ==="
  [ $rc -gt 128 ] && echo "   killed by signal $((rc-128)) (137=SIGKILL/OOM)"
  echo "   last stderr:"; tail -20 "$ERR" | sed 's/^/     /'; } >> "$ST"
