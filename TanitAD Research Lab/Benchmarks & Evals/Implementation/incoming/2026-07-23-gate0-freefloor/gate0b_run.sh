#!/bin/bash
# gate0b_run.sh -- REF-C-base + GRADIENT-NUDGE floor over the 38-scene suite in a clean logdir.
# Usage: bash gate0b_run.sh <logdir> <floorlog> <sceneset_dir>
# Renderer MUST be up on :6011. Runtime FOREGROUND; kills own controller/driver/physics by port.
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
PY="$ALPA/.venv/bin/python"
LOGDIR="$1"; FLOORLOG="$2"; SSDIR="$3"
TAG=$(basename "$LOGDIR")
NRE_HOST="$ALPA/data/nre-artifacts"
LOG=/workspace/${TAG}_run.log; : > "$LOG"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
cd "$ALPA" || { log "cd FAIL"; exit 2; }
: > "$FLOORLOG"
clean_ports(){ for p in 6789 6007 6006; do pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && kill "$pid" 2>/dev/null && log "killed :$p pid=$pid"; done; }

log "=== $TAG START floor=GRAD ==="
clean_ports; sleep 2
sed -i "s#/mnt/nre-data#$NRE_HOST#g" "$LOGDIR/generated-user-config-0.yaml"
sed -i 's|localhost:6005|localhost:6011|' "$LOGDIR/generated-network-config.yaml"
sed -i 's/allow_aggregation_with_failed_rollouts: false/allow_aggregation_with_failed_rollouts: true/' "$LOGDIR/eval-config.yaml"
SCENESET=$(grep -m1 'data_dir' "$LOGDIR/generated-user-config-0.yaml" | awk '{print $NF}' | tr -d '"')
mkdir -p "$LOGDIR/controller" /workspace/warp

setsid "$PY" -m alpasim_controller.server --port=6007 --log_dir="$LOGDIR/controller" \
  --log-level=INFO --config="$LOGDIR/controller-config.yaml" </dev/null >/workspace/${TAG}_controller.log 2>&1 &
log "controller launched"

DRVA="--port 6789 --ckpt /root/models/refc-base-30k/ckpt.pt --preset base --floor grad --lam 5.0 --mu 1.0 --clamp-m 0.75 --grad-eta 0.5 --grad-iters 4 --sceneset-dir $SSDIR --floor-log $FLOORLOG"
setsid env PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts:/workspace \
  "$PY" /workspace/refc_floor_driver.py $DRVA </dev/null >/workspace/${TAG}_driver.log 2>&1 &
log "driver launched: $DRVA"

export WARP_CACHE_PATH=/workspace/warp
setsid "$ALPA/.venv/bin/physics_server" --host=0.0.0.0 --port=6006 \
  --artifact-glob="$SCENESET/**/*.usdz" --use-ground-mesh=true --cache-size=16 --log-level=INFO \
  </dev/null >/workspace/${TAG}_physics.log 2>&1 &
log "physics launched"

UP=0
for i in $(seq 1 120); do
  UP=0; for p in 6006 6007 6789 6011; do ss -ltn | grep -q ":$p " && UP=$((UP+1)); done
  log "t=$((i*5))s ports_up=$UP/4"; [ "$UP" -ge 4 ] && break; sleep 5
done
if [ "$UP" -lt 4 ]; then log "ABORT ports $UP/4"; tail -30 /workspace/${TAG}_driver.log | sed 's/^/DRIVER: /' | tee -a "$LOG"; echo "FAIL ports=$UP" > "$LOGDIR/DONE"; exit 3; fi
log "driver up. tail:"; tail -5 /workspace/${TAG}_driver.log | sed 's/^/DRIVER: /' | tee -a "$LOG"

log "runtime START (38 scenes)"
"$PY" -m alpasim_runtime.simulate --user-config="$LOGDIR/generated-user-config-0.yaml" \
  --network-config="$LOGDIR/generated-network-config.yaml" --log-dir="$LOGDIR" --log-level=INFO \
  --array-job-dir="$LOGDIR" --eval-config="$LOGDIR/eval-config.yaml" >/workspace/${TAG}_runtime.log 2>&1
RT=$?
log "runtime EXIT=$RT"
clean_ports; sleep 2
AGG="$LOGDIR/aggregate/results-summary.json"
NROLL=$(grep -c 'rollout_id' "$AGG" 2>/dev/null || echo 0)
FEFF=$(grep -m1 'CANON f_eff' /workspace/${TAG}_driver.log 2>/dev/null || echo "no-feff")
NCLAMP=$(grep -c '"clamp": true' "$FLOORLOG" 2>/dev/null || echo 0)
log "aggregate=$([ -f "$AGG" ] && echo present || echo MISSING) n_rollout_id=$NROLL canon=$FEFF clamp_fires=$NCLAMP"
echo "DONE RT=$RT AGG=$([ -f "$AGG" ] && echo yes || echo no) NROLL=$NROLL CLAMP=$NCLAMP" > "$LOGDIR/DONE"
log "=== $TAG DONE (RT=$RT) ==="
