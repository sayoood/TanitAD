#!/bin/bash
# FULL-RUNTIME trafficsim reactivity experiment -- the follow-up GATE_RESULTS.md 2.4 names.
#
# Three arms x R repeats through `alpasim_runtime.simulate` with trafficsim=catk ENABLED:
#   GO   ego drives forward at 5 m/s   (simple_driver ConstantForwardPolicy)
#   STOP ego halts at handover         (same driver, --speed 0)
#   GO2  identical to GO               <- the stochastic-floor control
# Reaction => between(GO,STOP) - between(GO,GO2) positive and CI-separated.
#
# CONFIGURATION ONLY. No AlpaSim source is edited; the NuRec renderer is run, never modified
# (NGC-DL-CONTAINER-LICENSE forbids derivatives).
#
# Usage: bash ts_master.sh <scene_uuid> <n_sim_steps> <repeats>   (HF token on stdin line 1)
set -uo pipefail
IFS= read -r TOK
TOK="${TOK%$'\r'}"

SCENE="${1:?scene uuid}"
NSTEPS="${2:-150}"
R="${3:-3}"
ALPA=/workspace/alpa-invest/alpasim
PY="$ALPA/.venv/bin/python"
NRE_HOST="$ALPA/data/nre-artifacts"
USDZ="$NRE_HOST/all-usdzs"
BASE=/workspace/tsreact/$SCENE
mkdir -p "$BASE"

kill_port () { for p in "$@"; do
    pid=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -n "${pid:-}" ] && { echo "  killing PID $pid on :$p"; kill -9 "$pid" 2>/dev/null; }
  done; }

echo "############ SCENE $SCENE  steps=$NSTEPS repeats=$R"

# ---------------------------------------------------------------- 1. renderer (GPU, warm cache)
echo "=== renderer :6011 ==="
if ! ss -ltn | grep -q ":6011"; then
  export RUNFILES_DIR="/workspace/nre/rootfs/app/internal/scripts/pycena/runtime/pycena_nrm_full.runfiles"
  HOME=/workspace/nrehome XDG_CACHE_HOME=/tmp/.cache OMP_NUM_THREADS=1 \
  setsid /workspace/nre/rootfs/app/internal/scripts/pycena/runtime/pycena_nrm_full serve-grpc \
    --port=6011 --host=0.0.0.0 --artifact-glob="$USDZ/**/*.usdz" \
    --egocar-hood-dir="$NRE_HOST/ego-hoods" --no-enable-nrend \
    --download-cache-dir /tmp/nre-cache-dir --cache-size=5 --max-workers=4 --enable-editing-actors \
    </dev/null >/workspace/tsreact_renderer.log 2>&1 &
  echo "renderer pid=$!"
  for i in $(seq 1 90); do ss -ltn | grep -q ":6011" && break; sleep 5; done
fi
ss -ltn | grep -q ":6011" && echo "  :6011 UP" || { echo "RENDERER FAILED"; tail -20 /workspace/tsreact_renderer.log; exit 1; }

# ---------------------------------------------------------------- 2. arms
for ARM in GO STOP GO2; do
  case "$ARM" in
    GO|GO2) SPEED=5.0 ;;
    STOP)   SPEED=0.0 ;;
  esac
  for rep in $(seq 1 "$R"); do
    LOGDIR="$BASE/${ARM}_r${rep}"
    echo "======== ARM=$ARM rep=$rep speed=$SPEED -> $LOGDIR"
    [ -f "$LOGDIR/DONE" ] && { echo "  already done, skip"; continue; }

    # 2a. configs, trafficsim=catk
    printf '%s\n' "$TOK" | bash /workspace/ts_wizard_gen.sh "$SCENE" "$NSTEPS" "$LOGDIR" >"$LOGDIR.wizard.log" 2>&1
    [ -f "$LOGDIR/generated-user-config-0.yaml" ] || { echo "  WIZARD FAILED"; tail -20 "$LOGDIR.wizard.log"; continue; }

    # 2b. host paths + point renderer at :6011
    sed -i "s#/mnt/nre-data#$NRE_HOST#g" "$LOGDIR/generated-user-config-0.yaml"
    sed -i "s#/mnt/trafficsim-models#$ALPA/data/trafficsim-models#g" "$LOGDIR/trafficsim-config.yaml" 2>/dev/null
    sed -i 's|localhost:6005|localhost:6011|' "$LOGDIR/generated-network-config.yaml"
    sed -i 's|480|480|' "$LOGDIR/generated-user-config-0.yaml"

    kill_port 6006 6007 6008 6789
    mkdir -p "$LOGDIR/controller" /workspace/warp

    # 2c. trafficsim (CATK, CPU, :6007 -- the wizard SHIFTS ports when trafficsim is enabled -- torch_cluster here is a CPU-only source build, no nvcc on this pod)
    ( cd "$ALPA" && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 setsid "$ALPA/.venv/bin/catk_trafficsim_server" \
        --config-path="$LOGDIR" --config-name=trafficsim-config.yaml \
        server.port=6007 catk.device=cpu \
        catk.loader.usdz_folder="$USDZ" \
        </dev/null >"$LOGDIR/trafficsim.log" 2>&1 & echo "  trafficsim pid=$!" )

    # 2d. controller + driver + physics
    setsid "$PY" -m alpasim_controller.server --port=6008 --log_dir="$LOGDIR/controller" \
      --log-level=INFO --config="$LOGDIR/controller-config.yaml" \
      </dev/null >"$LOGDIR/controller.log" 2>&1 &
    setsid "$PY" /workspace/simple_driver.py --host 0.0.0.0 --port 6789 --speed "$SPEED" --hz 10 \
      </dev/null >"$LOGDIR/driver.log" 2>&1 &
    WARP_CACHE_PATH=/workspace/warp setsid "$ALPA/.venv/bin/physics_server" --host=0.0.0.0 --port=6006 \
      --artifact-glob="$USDZ/**/*.usdz" --use-ground-mesh=true --cache-size=16 \
      </dev/null >"$LOGDIR/physics.log" 2>&1 &

    echo -n "  waiting for ports: "
    for i in $(seq 1 60); do
      up=0; for p in 6006 6007 6008 6789; do ss -ltn | grep -q ":$p " && up=$((up+1)); done
      [ "$up" -ge 4 ] && break; sleep 5
    done
    for p in 6006 6007 6008 6789; do ss -ltn | grep -q ":$p " && echo -n "$p:UP " || echo -n "$p:DOWN "; done; echo

    # 2e. runtime, FOREGROUND so arms never overlap
    T0=$(date +%s)
    ( cd "$ALPA" && timeout 3600 "$PY" -m alpasim_runtime.simulate \
        --user-config="$LOGDIR/generated-user-config-0.yaml" \
        --network-config="$LOGDIR/generated-network-config.yaml" \
        --log-dir="$LOGDIR" --log-level=INFO --array-job-dir="$LOGDIR" \
        --eval-config="$LOGDIR/eval-config.yaml" </dev/null >"$LOGDIR/runtime.log" 2>&1 )
    RC=$?; T1=$(date +%s)
    echo "  runtime rc=$RC  wall=$((T1-T0))s"
    ls "$LOGDIR/rollouts/"*/*/rollout.asl >/dev/null 2>&1 && { echo "  ASL OK"; touch "$LOGDIR/DONE"; } \
      || { echo "  NO ASL"; tail -15 "$LOGDIR/runtime.log"; }
    grep -ci "error\|traceback" "$LOGDIR/trafficsim.log" 2>/dev/null | sed 's/^/  trafficsim errors: /'
    kill_port 6006 6007 6008 6789
  done
done

echo "=== cleanup: renderer left UP for the next scene; kill with kill_port 6011 ==="
echo "=== rollouts ==="
ls -d "$BASE"/*/rollouts/*/*/ 2>/dev/null | head -20
