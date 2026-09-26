#!/bin/bash
# ISOLATING ARM for the confounded real-vs-synthetic comparison.
#
# The first two runs differed in THREE variables at once (images, backbone, batch), so neither
# could be attributed. These two hold backbone (ViT-S) and batch (8) FIXED and move ONLY the images.
# ViT-S at batch 8 fits the 8 GB card; ViT-L does not, which is itself why the confound arose.
set -u
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
VPY="${REFE_DRIVERL_PY:-C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe}"
cd "$PKG/refe"
# Wait for the in-flight ViT-L real-image run to release the GPU. Two ViT-L/ViT-S jobs on an 8 GB
# card would spill to host RAM and make both timings meaningless -- the same trap as batch 4.
W="$PKG/raw/refe_overfit_realimages.txt"
for _ in $(seq 1 60); do
  grep -qE "TRAIN_CONTROL_(OK|FAILED)" "$W" 2>/dev/null && break
  sleep 20
done
echo "$(date +%T) GPU free (in-flight run finished: $(grep -oE 'TRAIN_CONTROL_[A-Z]+' "$W" 2>/dev/null | tail -1))"
echo "$(date +%T) === arm 1: ViT-S, batch 8, SYNTHETIC images ==="
"$VPY" -u train.py --overfit --overfit-n 8 --batch 8 --steps 300 --backbone vits16 --synthetic \
  --targets "D:/Projects/TanitAD/data/refe_targets_real" > "$PKG/raw/iso_vits_b8_synthetic.txt" 2>&1
echo "  arm1 exit=$?  $(tail -2 "$PKG/raw/iso_vits_b8_synthetic.txt" | head -1)"
echo "$(date +%T) === arm 2: ViT-S, batch 8, REAL images ==="
"$VPY" -u train.py --overfit --overfit-n 8 --batch 8 --steps 300 --backbone vits16 \
  --targets "D:/Projects/TanitAD/data/refe_targets_real" \
  --images "D:/Projects/TanitAD/data/nuplan-camera/scenarios" > "$PKG/raw/iso_vits_b8_real.txt" 2>&1
echo "  arm2 exit=$?  $(tail -2 "$PKG/raw/iso_vits_b8_real.txt" | head -1)"
echo "$(date +%T) IMAGE_ISOLATION_DONE"
