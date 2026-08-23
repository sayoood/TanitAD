#!/bin/bash
# rt_run.sh — instrumented closed-loop at 480x854 to measure the in-situ per-step LOOP period
# (render+physics+ipc+model). Ensures renderer up (LEFT up for reuse), runs the flagship rt_driver
# over the suite, writes per-step gap+model to /workspace/rt_timing.json. Physics on unless arg2=nophys.
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim; PY="$ALPA/.venv/bin/python"
SS=/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/e11a2e57085844fa5d905fa259abb344
GLOB="$SS/**/*.usdz"
LOGDIR="${1:-/workspace/rt_iso}"; MODE="${2:-phys}"; OUT="${3:-/workspace/rt_timing.json}"; KIND="${4:-flagship}"
ML=/workspace/rt_master.log; : > "$ML"
log(){ echo "[$(date -u +%H:%M:%S)] RT $*" | tee -a "$ML"; }
rm -f /workspace/rt_DONE
# renderer (leave up for reuse)
if ! ss -ltn | grep -q ':6011 '; then
  log "launch renderer"; setsid bash /workspace/renderer_serve.sh 6011 "$GLOB" </dev/null >/workspace/renderer_suite.log 2>&1 &
  for i in $(seq 1 90); do ss -ltn | grep -q ':6011 ' && grep -q 'Serving on 0.0.0.0:6011' /workspace/renderer_suite.log && break; sleep 5; done
fi
ss -ltn | grep -q ':6011 ' && log "renderer up" || { log "renderer FAIL"; exit 4; }
for p in 6789 6007 6006; do pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && kill "$pid" 2>/dev/null; done
sleep 2
rm -rf "$LOGDIR"; mkdir -p "$LOGDIR/controller" /workspace/warp
for f in controller-config.yaml driver-config.yaml eval-config.yaml generated-network-config.yaml generated-user-config-0.yaml run_metadata.yaml trafficsim-config.yaml wizard-config-loadable.yaml wizard-config.yaml; do cp /workspace/refcsuite/$f "$LOGDIR/$f"; done
sed -i "s#/mnt/nre-data#$ALPA/data/nre-artifacts#g" "$LOGDIR/generated-user-config-0.yaml"
sed -i 's|localhost:6005|localhost:6011|' "$LOGDIR/generated-network-config.yaml"
setsid "$PY" -m alpasim_controller.server --port=6007 --log_dir="$LOGDIR/controller" --config="$LOGDIR/controller-config.yaml" </dev/null >/workspace/rt_controller.log 2>&1 &
setsid env PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts "$PY" /workspace/rt_driver.py --port 6789 --kind "$KIND" --ckpt /root/models/flagship-30k/ckpt.pt --out "$OUT" </dev/null >/workspace/rt_driver_run.log 2>&1 &
NEED=3; PORTS="6007 6789 6011"
if [ "$MODE" = phys ]; then
  export WARP_CACHE_PATH=/workspace/warp
  setsid "$ALPA/.venv/bin/physics_server" --host=0.0.0.0 --port=6006 --artifact-glob="$GLOB" --use-ground-mesh=true --cache-size=16 </dev/null >/workspace/rt_physics.log 2>&1 &
  NEED=4; PORTS="6006 6007 6789 6011"
fi
for i in $(seq 1 72); do up=0; for p in $PORTS; do ss -ltn | grep -q ":$p " && up=$((up+1)); done; [ "$up" -ge "$NEED" ] && break; sleep 5; done
log "services up ($MODE); runtime start"
"$PY" -m alpasim_runtime.simulate --user-config="$LOGDIR/generated-user-config-0.yaml" --network-config="$LOGDIR/generated-network-config.yaml" --log-dir="$LOGDIR" --array-job-dir="$LOGDIR" --eval-config="$LOGDIR/eval-config.yaml" >/workspace/rt_runtime.log 2>&1
log "runtime exit=$?"
for p in 6789 6007 6006; do pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && kill "$pid" 2>/dev/null; done
echo "done" > /workspace/rt_DONE; log "DONE"
