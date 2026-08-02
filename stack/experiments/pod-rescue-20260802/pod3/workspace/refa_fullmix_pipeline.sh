#!/bin/bash
# pod3 REF-A-full-mix pipeline (self-chaining, survives session close):
# comma DINO precompute -> build mixed feature dir -> REF-A grid train (k=4).
# Idempotent-ish: skips comma precompute if already present.
set -e
PY=/workspace/venv/bin/python
LOG=/workspace/refa_fullmix.log
echo "[pipe] start $(date -u +%H:%M)" >> $LOG
cd /workspace/TanitAD/stack

# 1. comma DINO features (pai already done earlier)
if ! ls /workspace/dino_feats/comma2k19-train-*dinov2-b14/ep_*.pt >/dev/null 2>&1; then
  echo "[pipe] comma DINO precompute" >> $LOG
  $PY scripts/dino_precompute.py --cache-root /workspace/data \
    --out /workspace/dino_feats --train-n 411 --val-n 90 >> $LOG 2>&1
fi
echo "[pipe] comma feats: $(ls /workspace/dino_feats/comma2k19-train-*dinov2-b14/ | wc -l)" >> $LOG

# 2. build mixed feature dir (symlink comma + pai x2 for ~0.6 pai share)
CT=$(ls -d /workspace/dino_feats/comma2k19-train-*dinov2-b14 | head -1)
CV=$(ls -d /workspace/dino_feats/comma2k19-val-*dinov2-b14 | head -1)
PT=$(ls -d /workspace/dino_feats/physicalai-train-*dinov2-b14 | head -1)
PV=$(ls -d /workspace/dino_feats/physicalai-val-*dinov2-b14 | head -1)
MT=/workspace/dino_feats/mix-train-v1; MV=/workspace/dino_feats/mix-val-v1
rm -rf "$MT" "$MV"; mkdir -p "$MT" "$MV"
i=0; for f in "$CT"/ep_*.pt; do ln -s "$f" "$MT/ep_c$(printf %05d $i).pt"; i=$((i+1)); done
i=0; for f in "$PT"/ep_*.pt; do ln -s "$f" "$MT/ep_p$(printf %05d $i)a.pt"; ln -s "$f" "$MT/ep_p$(printf %05d $i)b.pt"; i=$((i+1)); done
i=0; for f in "$CV"/ep_*.pt; do ln -s "$f" "$MV/ep_c$(printf %05d $i).pt"; i=$((i+1)); done
i=0; for f in "$PV"/ep_*.pt; do ln -s "$f" "$MV/ep_p$(printf %05d $i).pt"; i=$((i+1)); done
echo "[pipe] mix train=$(ls $MT|wc -l) val=$(ls $MV|wc -l)" >> $LOG

# 3. REF-A grid-adapter full-mix train (feature-level; no OOM risk)
echo "[pipe] launch REF-A-fullmix $(date -u +%H:%M)" >> $LOG
$PY scripts/refa_train.py --data-root /workspace/dino_feats \
  --out /workspace/experiments/refa-fullmix-30k --steps 30000 \
  --rollout-k 4 --batch 64 --adapter grid \
  > /workspace/experiments/refa-fullmix-30k.log 2>&1
echo "[pipe] REF-A-fullmix done $(date -u +%H:%M)" >> $LOG
