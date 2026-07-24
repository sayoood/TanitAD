#!/bin/bash
# Start backing services + REF-C driver for the 12-scene SUITE run. Renderer assumed up on :6011.
# Usage: bash refc_suite_launch.sh [ckpt] [preset]
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
cd "$ALPA" || exit 1
PY="$ALPA/.venv/bin/python"
M2=/workspace/refcsuite
SCENESET="$ALPA/data/nre-artifacts/scenesets/e11a2e57085844fa5d905fa259abb344"
CKPT="${1:-/root/models/refc-base-30k/ckpt.pt}"
PRESET="${2:-base}"
mkdir -p "$M2/controller" /workspace/warp
sed -i 's|localhost:6005|localhost:6011|' "$M2/generated-network-config.yaml"

setsid "$PY" -m alpasim_controller.server --port=6007 --log_dir="$M2/controller" \
  --log-level=INFO --config="$M2/controller-config.yaml" \
  </dev/null >/workspace/suite_controller.log 2>&1 &
echo "controller pid=$!"

setsid env PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts "$PY" \
  /workspace/refc_driver.py --port 6789 --ckpt "$CKPT" --preset "$PRESET" \
  </dev/null >/workspace/suite_driver.log 2>&1 &
echo "refc_driver pid=$! (ckpt=$CKPT preset=$PRESET)"

export WARP_CACHE_PATH=/workspace/warp
setsid "$ALPA/.venv/bin/physics_server" --host=0.0.0.0 --port=6006 \
  --artifact-glob="$SCENESET/**/*.usdz" --use-ground-mesh=true --cache-size=16 --log-level=INFO \
  </dev/null >/workspace/suite_physics.log 2>&1 &
echo "physics pid=$!"

for i in $(seq 1 40); do
  up=0; for p in 6006 6007 6789 6011; do ss -ltn | grep -q ":$p" && up=$((up+1)); done
  echo "t=$((i*5))s ports_up=$up/4"; [ "$up" -ge 4 ] && break; sleep 5
done
tail -2 /workspace/suite_driver.log
