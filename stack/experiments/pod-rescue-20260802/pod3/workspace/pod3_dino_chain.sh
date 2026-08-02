#!/bin/bash
# pod3: wait for the corrected epcache build to finish, then regenerate the
# PhysicalAI DINO features from the CORRECTED frames. comma feats untouched.
LOG=/workspace/dino_chain.log
echo "[chain] start $(date)" > "$LOG"

# 1) wait for epcache PAI_CACHE_DONE (or die if the build process vanishes)
while ! grep -q PAI_CACHE_DONE /workspace/build_pai.log 2>/dev/null; do
  if ! pgrep -f build_pai_cache >/dev/null 2>&1; then
    sleep 3
    grep -q PAI_CACHE_DONE /workspace/build_pai.log 2>/dev/null && break
    echo "[chain] ERROR: epcache build gone without PAI_CACHE_DONE $(date)" >> "$LOG"
    exit 1
  fi
  sleep 20
done
echo "[chain] epcache done $(date)" >> "$LOG"
ls -1d /workspace/data/physicalai/_epcache/*/ >> "$LOG" 2>&1

# 2) move the STALE (wrong-zoom) physicalai DINO dirs out of the glob path;
#    keep comma2k19-*-dinov2-b14 exactly as-is.
mkdir -p /workspace/dino_feats_old
mv /workspace/dino_feats/physicalai-*-dinov2-b14 /workspace/dino_feats_old/ 2>/dev/null
echo "[chain] stale physicalai dino moved; remaining feats:" >> "$LOG"
ls -1d /workspace/dino_feats/*/ >> "$LOG" 2>&1

# 3) regenerate physicalai DINO features from the corrected epcache
cd /workspace/TanitAD/stack
echo "[chain] launching dino_precompute $(date)" >> "$LOG"
/workspace/venv/bin/python scripts/dino_precompute.py \
  --cache-root /workspace/data/physicalai/_epcache \
  --out /workspace/dino_feats --train-n 402 --val-n 101 >> "$LOG" 2>&1
echo "[chain] DINO_CHAIN_DONE $(date)" >> "$LOG"
