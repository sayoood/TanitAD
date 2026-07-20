#!/bin/bash
# REF-B 20h supervisor (pod2, survives session close). Keeps the trainer AND
# the OOM guard alive; auto-resumes REF-B from ckpt.pt on any death; exits at
# 30k. NEVER double-launches (pgrep-guarded) so it can't disrupt a healthy run.
LOG=/workspace/experiments/refb-30k.log
OUT=/workspace/experiments/refb-30k
cd /workspace/TanitAD/stack
while true; do
  # guard
  if ! pgrep -f "cache_guard_v[2]" >/dev/null; then
    setsid bash /workspace/cache_guard_v2.sh > /workspace/guard_v2.log 2>&1 < /dev/null &
    echo "[sup] guard relaunched $(date -u +%H:%M)" >> /workspace/refb_sup.log
  fi
  STEP=$(grep -o '"step": [0-9]*' "$LOG" 2>/dev/null | tail -1 | grep -o '[0-9]*')
  [ -z "$STEP" ] && STEP=0
  if grep -q '"done": true' "$LOG" 2>/dev/null || [ "$STEP" -ge 29999 ]; then
    echo "[sup] REF-B complete at step $STEP $(date -u +%H:%M) — supervisor exit" >> /workspace/refb_sup.log
    exit 0
  fi
  if ! pgrep -f "refb_trai[n]" >/dev/null; then
    echo "[sup] REF-B DOWN at step $STEP — resuming $(date -u +%H:%M)" >> /workspace/refb_sup.log
    setsid python scripts/refb_train.py --data-root /workspace/data/mix/_epcache \
      --out "$OUT" --steps 30000 --batch 12 --grad-checkpoint >> "$LOG" 2>&1 < /dev/null &
    sleep 30
  fi
  sleep 180
done
