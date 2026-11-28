#!/bin/bash
# Launch the NRE/pycena NuRec renderer bare (no container) on tanitad-eval.
# Serves the gRPC sensorsim interface on --port for the given scene USDZ(s).
# Requires: /app symlink -> /workspace/nre/rootfs/app (belt+suspenders for absolute runfiles paths).
# Caches -> /workspace (/ is 93% full). Usage: bash renderer_serve.sh <port> <scene_glob> [ego_hoods_dir]
set -uo pipefail
PORT="${1:-6011}"
SCENE_GLOB="${2:-/workspace/scene_dl/sample_set/26.04_release/**/*.usdz}"
EGO_HOODS="${3:-/workspace/alpa-invest/alpasim/data/nre-artifacts/ego-hoods}"
ROOTFS=/workspace/nre/rootfs
BIN="$ROOTFS/app/internal/scripts/pycena/runtime/pycena_nrm_full"
export RUNFILES_DIR="$BIN.runfiles"
export HOME=/workspace/nrehome
export XDG_CACHE_HOME=/workspace/nrehome/.cache
export OMP_NUM_THREADS=1
export NVIDIA_DRIVER_CAPABILITIES=all
mkdir -p "$HOME" "$XDG_CACHE_HOME" /workspace/nre-cache
echo "RENDERER_LAUNCH port=$PORT $(date -u +%H:%M:%S)"
echo "glob=$SCENE_GLOB ego=$EGO_HOODS RUNFILES_DIR=$RUNFILES_DIR"
exec "$BIN" serve-grpc \
  --port="$PORT" \
  --host=0.0.0.0 \
  --artifact-glob="$SCENE_GLOB" \
  --egocar-hood-dir="$EGO_HOODS" \
  --no-enable-nrend \
  --download-cache-dir=/workspace/nre-cache \
  --cache-size=5 \
  --max-workers=4 \
  --enable-editing-actors
