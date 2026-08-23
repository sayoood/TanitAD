#!/bin/bash
# gate1_run.sh -- run ONE driver over the junction sceneset in an isolated logdir,
# WITH per-step (pose, speed, rig-frame plan) logging to <logdir>/preds.jsonl (--log-preds).
# Derived from vs_suite_run.sh (validated M2 topology); the ONLY change is the driver gains
# --log-preds so we capture the on-policy visited-state + issued-plan stream in CLOSED loop
# (not force-GT). Renderer MUST already be up on :6011 (shared, never killed here).
# Usage: bash gate1_run.sh <logdir> <driver_py> <ckpt> [preset]
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
PY="$ALPA/.venv/bin/python"
LOGDIR="$1"; DRIVER="$2"; CKPT="$3"; PRESET="${4:-}"
TAG=$(basename "$LOGDIR")
NRE_HOST="$ALPA/data/nre-artifacts"
LOG=/workspace/${TAG}_run.log
PREDS="$LOGDIR/preds.jsonl"
: > "$LOG"
: > "$PREDS"                 # fresh per-step log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
cd "$ALPA" || { log "cd FAIL"; exit 2; }

clean_ports(){
  for p in 6789 6007 6006; do
    pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -n "$pid" ] && kill "$pid" 2>/dev/null && log "killed :$p pid=$pid"
  done
}

log "=== $TAG START (with --log-preds $PREDS) ==="
clean_ports; sleep 2

# idempotent config rewrites (container->host paths + renderer port)
sed -i "s#/mnt/nre-data#$NRE_HOST#g" "$LOGDIR/generated-user-config-0.yaml"
sed -i 's|localhost:6005|localhost:6011|' "$LOGDIR/generated-network-config.yaml"
SCENESET=$(grep -m1 'data_dir' "$LOGDIR/generated-user-config-0.yaml" | awk '{print $NF}' | tr -d '"')
log "driver=$DRIVER ckpt=$CKPT preset='$PRESET' sceneset=$SCENESET"
mkdir -p "$LOGDIR/controller" /workspace/warp

setsid "$PY" -m alpasim_controller.server --port=6007 --log_dir="$LOGDIR/controller" \
  --log-level=INFO --config="$LOGDIR/controller-config.yaml" \
  </dev/null >/workspace/${TAG}_controller.log 2>&1 &
log "controller launched"

if [ -n "$PRESET" ]; then DRVA="--port 6789 --ckpt $CKPT --preset $PRESET --log-preds $PREDS";
else DRVA="--port 6789 --ckpt $CKPT --log-preds $PREDS"; fi
setsid env PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts "$PY" "$DRIVER" $DRVA \
  </dev/null >/workspace/${TAG}_driver.log 2>&1 &
log "driver launched: $DRIVER $DRVA"

export WARP_CACHE_PATH=/workspace/warp
setsid "$ALPA/.venv/bin/physics_server" --host=0.0.0.0 --port=6006 \
  --artifact-glob="$SCENESET/**/*.usdz" --use-ground-mesh=true --cache-size=16 --log-level=INFO \
  </dev/null >/workspace/${TAG}_physics.log 2>&1 &
log "physics launched"

UP=0
for i in $(seq 1 96); do
  UP=0; for p in 6006 6007 6789 6011; do ss -ltn | grep -q ":$p " && UP=$((UP+1)); done
  log "t=$((i*5))s ports_up=$UP/4"
  [ "$UP" -ge 4 ] && break
  sleep 5
done
if [ "$UP" -lt 4 ]; then
  log "ABORT: ports not up ($UP/4)"
  tail -25 /workspace/${TAG}_driver.log | sed 's/^/DRIVER: /' | tee -a "$LOG"
  echo "FAIL ports=$UP" > "$LOGDIR/DONE"; exit 3
fi
log "driver up. tail:"; tail -4 /workspace/${TAG}_driver.log | sed 's/^/DRIVER: /' | tee -a "$LOG"

NSC=$(grep -c 'scene_id:' "$LOGDIR/generated-user-config-0.yaml")
log "runtime START ($NSC scenes)"
"$PY" -m alpasim_runtime.simulate \
  --user-config="$LOGDIR/generated-user-config-0.yaml" \
  --network-config="$LOGDIR/generated-network-config.yaml" \
  --log-dir="$LOGDIR" --log-level=INFO --array-job-dir="$LOGDIR" \
  --eval-config="$LOGDIR/eval-config.yaml" >/workspace/${TAG}_runtime.log 2>&1
RT=$?
log "runtime EXIT=$RT"

clean_ports; sleep 2
AGG="$LOGDIR/aggregate/results-summary.json"
NROLL=$(ls -1 "$LOGDIR"/rollouts/clipgt-*/*/metrics.parquet 2>/dev/null | wc -l)
NPRED=$(wc -l < "$PREDS" 2>/dev/null || echo 0)
FEFF=$(grep -m1 'CANON f_eff' /workspace/${TAG}_driver.log 2>/dev/null || echo "no-feff-line")
log "rollouts_with_parquet=$NROLL preds_lines=$NPRED"
log "driver canon: $FEFF"
echo "DONE RT=$RT NROLL=$NROLL NPRED=$NPRED AGG=$([ -f "$AGG" ] && echo yes || echo no)" > "$LOGDIR/DONE"
log "=== $TAG DONE (RT=$RT) ==="
