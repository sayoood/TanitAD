#!/bin/bash
# Start services + REF-C base driver (with --log-preds) for the FORCE-GT open-loop diagnostic.
# Renderer assumed up on :6011 (on the openloop sceneset).
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
cd "$ALPA" || exit 1
PY="$ALPA/.venv/bin/python"
M2=/workspace/refcopenloop
SCENESET="$ALPA/data/nre-artifacts/scenesets/4aa6a3b8a03d548fa84ebae65c83127c"
mkdir -p "$M2/controller" /workspace/warp
rm -f /workspace/refc_openloop_preds.jsonl
sed -i 's|localhost:6005|localhost:6011|' "$M2/generated-network-config.yaml"

setsid "$PY" -m alpasim_controller.server --port=6007 --log_dir="$M2/controller" \
  --log-level=INFO --config="$M2/controller-config.yaml" \
  </dev/null >/workspace/ol_controller.log 2>&1 &
echo "controller pid=$!"

setsid env PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts "$PY" \
  /workspace/refc_driver.py --port 6789 --ckpt /root/models/refc-base-30k/ckpt.pt --preset base \
  --log-preds /workspace/refc_openloop_preds.jsonl \
  </dev/null >/workspace/ol_driver.log 2>&1 &
echo "refc_driver pid=$!"

export WARP_CACHE_PATH=/workspace/warp
setsid "$ALPA/.venv/bin/physics_server" --host=0.0.0.0 --port=6006 \
  --artifact-glob="$SCENESET/**/*.usdz" --use-ground-mesh=true --cache-size=16 --log-level=INFO \
  </dev/null >/workspace/ol_physics.log 2>&1 &
echo "physics pid=$!"

for i in $(seq 1 40); do
  up=0; for p in 6006 6007 6789 6011; do ss -ltn | grep -q ":$p" && up=$((up+1)); done
  echo "t=$((i*5))s ports_up=$up/4"; [ "$up" -ge 4 ] && break; sleep 5
done
tail -2 /workspace/ol_driver.log
