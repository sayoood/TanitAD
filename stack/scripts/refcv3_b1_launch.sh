#!/bin/bash
# REF-C v3 — the B1 + v7.2 launch chain (PI directive 2026-09-02).
#
# ⛔ EVERY STEP IS A VERIFY-GATE. The programme's rule is that a chain must
# REFUSE to run on a precondition it has not checked, because the failures that
# cost days were all "it ran, on the wrong thing": a stale pod checkout that
# resurrected a fixed bug, a 48-worker build that deadlocked while still
# looking alive, an 8-wide head trained on 3 classes through a passing
# preflight. Each gate below prints what it checked.
#
# ⛔ NO GIT SYNC IN THIS CHAIN. Pods have no git credentials — `git fetch`
# HANGS, and a checkout after a failed fetch RESETS the tree to an ancient
# commit, destroying shipped files. Code arrives by file-ship, verified here.
set -uo pipefail

OUT=/workspace/experiments/refcv3-b1-v72-30k
CACHE=/workspace/TanitAD/data/b1-epcache-4719
LABELS=/workspace/TanitAD/data/s2_labels_v7.2_train.jsonl.gz
STACK=/workspace/TanitAD/stack
LOG=/workspace/refcv3_launch.log
exec >> "$LOG" 2>&1
echo "=== chain start $(date -u +%FT%TZ) ==="

# --- gate 1: the corpus is COMPLETE ---------------------------------------
# 4,713 is the parity-gated count (6 val40 clips dropped); never 4,719.
WANT=4713
while true; do
  N=$(ls "$CACHE"/*.v2ep.pt 2>/dev/null | wc -l)
  ALIVE=$(ps -eo args | grep -c "[p]od_build_b1_epcache")
  if [ "$N" -ge "$WANT" ]; then echo "[gate1] corpus complete: $N"; break; fi
  if [ "$ALIVE" -eq 0 ]; then
    echo "[gate1] ⛔ REFUSING: builder is GONE at $N/$WANT episodes."
    echo "[gate1]   A partial corpus is NOT B1 — 30k steps on a subset is a"
    echo "[gate1]   different experiment, not an early one. Restart the build."
    exit 2
  fi
  sleep 120
done

# --- gate 2: the shipped code carries the launch features ------------------
# grep-verify the FIX IS PRESENT rather than trusting that a scp happened.
for pat in "v7-labels" "nav_inject" "uplink_grad" "window_in_band"; do
  if ! grep -rq "$pat" "$STACK/scripts/refc_v3_train.py" \
        "$STACK/tanitad/refs/refc_v3.py" "$STACK/tanitad/data/v7_labels.py"; then
    echo "[gate2] ⛔ REFUSING: '$pat' absent from the shipped stack — this pod"
    echo "[gate2]   is running code older than the launch config."
    exit 2
  fi
done
echo "[gate2] shipped stack carries: v7-labels, nav_inject, uplink_grad, window_in_band"

# --- gate 3: the labels are the CANONICAL release --------------------------
MD5=$(md5sum "$LABELS" | cut -d' ' -f1)
if [ "$MD5" != "0ff902130ce76886b8a925eceed9e3a5" ]; then
  echo "[gate3] ⛔ REFUSING: label md5 $MD5 is not the canonical v7.2 train release."
  exit 2
fi
echo "[gate3] labels md5 OK (canonical v7.2 train, 4,572 records)"

# --- gate 4: preflight AT THE LAUNCH GEOMETRY, with the launch labels -------
ulimit -n 65536
export OMP_NUM_THREADS=2
cd /workspace/TanitAD
if ! PYTHONPATH="$STACK" python3 "$STACK/scripts/refc_v3_train.py" \
      --arm hier --preflight --image-hw 256 640 --v7-labels "$LABELS" \
      --out /tmp/v3pf_launch; then
  echo "[gate4] ⛔ REFUSING: preflight FAILED at the launch geometry."
  exit 2
fi
echo "[gate4] preflight PASS"

# --- launch ----------------------------------------------------------------
# workers 4: MEASURED 3.32x over workers=0 (6.30 -> 1.90 s/step). Higher counts
# died on the container's 1024-fd soft limit; `ulimit -n` above lifts that, but
# the pod's CPU quota is 7.65 (cfs_quota 765000/100000, NOT the 96 nproc
# reports), so more workers cannot buy throughput anyway.
# ⭐ size base (PI override 2026-09-02): 106,847,621 params. MEASURED peak
# 35.91 GB of the A40's 44.3 (81 %) at batch 20 -- it fits, with less headroom
# than small's 26.85 GB (61 %), so an OOM here is a real risk to watch rather
# than an impossibility. batch 20 is the registered value and is KEPT so the
# sample budget (30k x 20 = 600k) matches the registered arm.
# ⚠️ base exceeds the v3 design's "<= 80 M hard" budget by 33 %; see
# GOALS_AND_CLAIMS D-REFCV3-SIZE for the override and the contrary measurement.
mkdir -p "$OUT"
echo "[launch] $(date -u +%FT%TZ) starting 30k"
PYTHONPATH="$STACK" nohup python3 -u "$STACK/scripts/refc_v3_train.py" \
  --arm hier --size base \
  --v2-cache "$CACHE" \
  --v7-labels "$LABELS" \
  --image-hw 256 640 \
  --steps 30000 --batch 20 --workers 4 --v2-lru 6 \
  --lr 1e-4 --warmup 2000 --seed 0 \
  --log-every 50 --save-every 500 \
  --out "$OUT" >> "$OUT/train.log" 2>&1 &
echo "[launch] pid $! -> $OUT"
