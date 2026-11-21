#!/bin/bash
# Start the bare AlpaSim backing services for a localhost closed-loop run on tanitad-eval.
# Renderer is assumed ALREADY running on :6011 (warm). Points the wizard network-config at it.
# Starts controller(:6007), our simple_driver(:6789), physics(:6006) detached. Then waits for ports.
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
cd "$ALPA" || exit 1
PY="$ALPA/.venv/bin/python"
SCENESET="$ALPA/data/nre-artifacts/scenesets/482d8796dfba79cc76c7b1f759e3d6b1"
mkdir -p /workspace/m2run/controller /workspace/warp

# 1. Point the runtime's renderer endpoint at the warm renderer on :6011.
sed -i 's|localhost:6005|localhost:6011|' /workspace/m2run/generated-network-config.yaml
echo "renderer endpoint ->"; grep -A1 "^renderer:" /workspace/m2run/generated-network-config.yaml

# 2. controller (:6007, CPU)
setsid "$PY" -m alpasim_controller.server --port=6007 --log_dir=/workspace/m2run/controller \
  --log-level=INFO --config=/workspace/m2run/controller-config.yaml \
  </dev/null >/workspace/controller.log 2>&1 &
echo "controller pid=$!"

# 3. our external driver (:6789, CPU)
setsid "$PY" /workspace/simple_driver.py --host 0.0.0.0 --port 6789 --speed 5 --hz 10 \
  </dev/null >/workspace/driver.log 2>&1 &
echo "driver pid=$!"

# 4. physics (:6006, GPU; warp JIT on first run)
export WARP_CACHE_PATH=/workspace/warp
setsid "$ALPA/.venv/bin/physics_server" --host=0.0.0.0 --port=6006 \
  --artifact-glob="$SCENESET/**/*.usdz" --use-ground-mesh=true --cache-size=16 --log-level=INFO \
  </dev/null >/workspace/physics.log 2>&1 &
echo "physics pid=$!"

echo "=== waiting up to 200s for ports 6006 6007 6789 (renderer 6011 already up) ==="
for i in $(seq 1 40); do
  up=0
  for p in 6006 6007 6789 6011; do ss -ltn | grep -q ":$p" && up=$((up+1)); done
  echo "t=$((i*5))s ports_up=$up/4"
  [ "$up" -ge 4 ] && break
  sleep 5
done
echo "=== final port state ==="
for p in 6005 6006 6007 6011 6789; do ss -ltn | grep -q ":$p" && echo "  :$p UP" || echo "  :$p down"; done
