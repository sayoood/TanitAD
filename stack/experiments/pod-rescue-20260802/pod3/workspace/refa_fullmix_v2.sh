#!/bin/bash
# pod3 REF-A-full-mix pipeline v2 — hardened against the *train*/*val* glob trap.
# Resumable comma precompute + COUNT GUARD + mix isolated in its own parent.
set -e
PY=/workspace/venv/bin/python
LOG=/workspace/refa_fullmix.log
echo "[pipe v2] start $(date -u +%H:%M)" >> $LOG
cd /workspace/TanitAD/stack

# 1. comma DINO features to completion (idempotent — skips already-done eps)
$PY scripts/dino_precompute.py --cache-root /workspace/data \
    --out /workspace/dino_feats --train-n 410 --val-n 90 >> $LOG 2>&1
CN=$(ls /workspace/dino_feats/comma2k19-train-*dinov2-b14/ep_*.pt 2>/dev/null | wc -l)
echo "[pipe v2] comma train feats: $CN" >> $LOG
if [ "$CN" -lt 400 ]; then echo "[pipe v2] ABORT: comma feats $CN < 400" >> $LOG; exit 1; fi

# 2. mix in a DEDICATED parent (only mix-train/mix-val — no stray *train* dirs)
CT=$(ls -d /workspace/dino_feats/comma2k19-train-*dinov2-b14 | head -1)
CV=$(ls -d /workspace/dino_feats/comma2k19-val-*dinov2-b14 | head -1)
PT=$(ls -d /workspace/dino_feats/physicalai-train-*dinov2-b14 | head -1)
PV=$(ls -d /workspace/dino_feats/physicalai-val-*dinov2-b14 | head -1)
MROOT=/workspace/mixfeats
MT=$MROOT/mix-train-v1; MV=$MROOT/mix-val-v1
rm -rf "$MROOT"; mkdir -p "$MT" "$MV"
i=0; for f in "$CT"/ep_*.pt; do ln -s "$f" "$MT/ep_c$(printf %05d $i).pt"; i=$((i+1)); done
i=0; for f in "$PT"/ep_*.pt; do ln -s "$f" "$MT/ep_p$(printf %05d $i)a.pt"; ln -s "$f" "$MT/ep_p$(printf %05d $i)b.pt"; i=$((i+1)); done
i=0; for f in "$CV"/ep_*.pt; do ln -s "$f" "$MV/ep_c$(printf %05d $i).pt"; i=$((i+1)); done
i=0; for f in "$PV"/ep_*.pt; do ln -s "$f" "$MV/ep_p$(printf %05d $i).pt"; i=$((i+1)); done
NT=$(ls "$MT" | wc -l); NV=$(ls "$MV" | wc -l)
echo "[pipe v2] mix train=$NT val=$NV" >> $LOG
if [ "$NT" -lt 1000 ]; then echo "[pipe v2] ABORT: mix train $NT too small" >> $LOG; exit 1; fi

# 3. REF-A grid full-mix train (feature-level; --data-root = the mix-only parent)
echo "[pipe v2] launch REF-A-fullmix $(date -u +%H:%M)" >> $LOG
$PY scripts/refa_train.py --data-root $MROOT \
    --out /workspace/experiments/refa-fullmix-30k --steps 30000 \
    --rollout-k 4 --batch 64 --adapter grid \
    >> /workspace/experiments/refa-fullmix-30k.log 2>&1
echo "[pipe v2] REF-A-fullmix done $(date -u +%H:%M)" >> $LOG
