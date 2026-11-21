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
# ⛔ THE GATE WAITS ON PROGRESS, NOT ON LIVENESS. MEASURED TWICE 2026-09-02:
# the builder stalled with every worker dead and the parent parked in
# futex_wait_queue — still "alive" to any ps check, log frozen for ~2 h. A
# liveness gate would have waited forever on that; only the FILE COUNT moving
# distinguishes building from hung.
LAST=$(ls "$CACHE"/*.v2ep.pt 2>/dev/null | wc -l)
LASTT=$(date +%s)
STALL_LIMIT=1800
while true; do
  N=$(ls "$CACHE"/*.v2ep.pt 2>/dev/null | wc -l)
  if [ "$N" -ge "$WANT" ]; then echo "[gate1] corpus complete: $N"; break; fi
  if [ "$N" -gt "$LAST" ]; then LAST=$N; LASTT=$(date +%s); fi
  SUP=$(ps -eo args | grep -c "[s]up_b1build")
  if [ $(( $(date +%s) - LASTT )) -ge "$STALL_LIMIT" ]; then
    echo "[gate1] ⛔ REFUSING: no new episode in ${STALL_LIMIT}s (at $N/$WANT,"
    echo "[gate1]   supervisor procs=$SUP). A partial corpus is NOT B1 — 30k"
    echo "[gate1]   steps on a subset is a DIFFERENT experiment, not an early"
    echo "[gate1]   one. Fix the build; this chain will not launch on it."
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

# --- gate 5: SPLIT the corpus, and refuse to train on the eval clips --------
# MEASURED LEAK: the v7.2 release is 4,572 train / 147 eval, disjoint -- but
# raw B1 is 4,713 = 4,572 + 141 of those eval clips. Training on 'all of B1'
# trains on 141 of the 147 eval clips, and the leak would not announce itself:
# the eval would simply look good.
TRAIN_DIR=/workspace/TanitAD/data/b1-train-v72
EVAL_DIR=/workspace/TanitAD/data/b1-eval-v72
EVAL_LABELS=/workspace/TanitAD/data/s2_labels_v7.2_eval.jsonl.gz
if ! PYTHONPATH="$STACK" python3 "$STACK/scripts/refcv3_make_split.py" \
      --cache "$CACHE" --train-labels "$LABELS" --eval-labels "$EVAL_LABELS" \
      --out-train "$TRAIN_DIR" --out-eval "$EVAL_DIR"; then
  echo "[gate5] REFUSING: could not build a disjoint train/eval split."
  exit 2
fi
echo "[gate5] split built and verified disjoint"

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
#
# ⭐ steps 40284 = ONE FULL EPOCH (PI 2026-09-02), not a wall-clock target.
#   MEASURED: 170.95 windows/episode (60-episode probe: 10,257 windows) x 4,713
#   episodes = 805,687 windows; / batch 20 = 40,284 steps. Was 30,000 = 0.74
#   epochs, which the design chose deliberately ("still under one epoch").
# ⛔ 40,284 IS NOT A PORTABLE CONSTANT. It is corpus-AND-batch-specific: one
#   epoch is windows/batch, so at batch 8 on this corpus it is 100,711, and on
#   the 2,400-episode parity corpus (415,002 windows) it is 51,875 at batch 8.
#   Copying "40284" onto another arm would be the derived-constant-out-of-scope
#   error this programme has paid for repeatedly. Recompute per arm.
mkdir -p "$OUT"
echo "[launch] $(date -u +%FT%TZ) starting 30k"
PYTHONPATH="$STACK" nohup python3 -u "$STACK/scripts/refc_v3_train.py" \
  --arm hier --size base \
  --v2-cache "$TRAIN_DIR" \
  --v7-labels "$LABELS" \
  --eval-cache "$EVAL_DIR" --eval-labels "$EVAL_LABELS" \
  --eval-every 100 --eval-batches 8 \
  --image-hw 256 640 \
  --steps 40284 --batch 20 --workers 4 --v2-lru 6 \
  --lr 1e-4 --warmup 2000 --seed 0 \
  --log-every 50 --save-every 500 \
  --out "$OUT" >> "$OUT/train.log" 2>&1 &
echo "[launch] pid $! -> $OUT"
