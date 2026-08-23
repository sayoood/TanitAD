#!/bin/bash
# Start services + flagship v1 driver (flagship-30k tactical policy) for a single-scene
# closed-loop on clip 01d503d4 (refcrun config, native 1080x1920). Renderer up on :6011.
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
cd "$ALPA" || exit 1
PY="$ALPA/.venv/bin/python"
M2=/workspace/refcrun
SCENESET="$ALPA/data/nre-artifacts/scenesets/482d8796dfba79cc76c7b1f759e3d6b1"
mkdir -p "$M2/controller" /workspace/warp
sed -i 's|localhost:6005|localhost:6011|' "$M2/generated-network-config.yaml"

setsid "$PY" -m alpasim_controller.server --port=6007 --log_dir="$M2/controller" \
  --log-level=INFO --config="$M2/controller-config.yaml" \
  </dev/null >/workspace/fl_controller.log 2>&1 &
echo "controller pid=$!"

setsid env PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts "$PY" \
  /workspace/flagship_v1_driver.py --port 6789 --ckpt /root/models/flagship-30k/ckpt.pt \
  </dev/null >/workspace/fl_driver.log 2>&1 &
echo "flagship_driver pid=$!"

export WARP_CACHE_PATH=/workspace/warp
setsid "$ALPA/.venv/bin/physics_server" --host=0.0.0.0 --port=6006 \
  --artifact-glob="$SCENESET/**/*.usdz" --use-ground-mesh=true --cache-size=16 --log-level=INFO \
  </dev/null >/workspace/fl_physics.log 2>&1 &
echo "physics pid=$!"

for i in $(seq 1 40); do
  up=0; for p in 6006 6007 6789 6011; do ss -ltn | grep -q ":$p" && up=$((up+1)); done
  echo "t=$((i*5))s ports_up=$up/4"; [ "$up" -ge 4 ] && break; sleep 5
done
tail -2 /workspace/fl_driver.log
