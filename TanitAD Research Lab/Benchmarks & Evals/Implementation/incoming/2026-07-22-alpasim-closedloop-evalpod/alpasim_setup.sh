#!/bin/bash
# AlpaSim bare workspace setup on tanitad-eval. All caches -> /workspace (/ is 93% full).
# Excludes the driver member (skips heavy vam/alpamayo git deps); we run our own driver.
# Idempotent-ish; logs to stdout (redirected by caller). Markers: STAGE_*, SETUP_DONE / SETUP_FAIL.
set -uo pipefail
ALPA=/workspace/alpa-invest/alpasim
export CARGO_HOME=/workspace/.cargo RUSTUP_HOME=/workspace/.rustup
export UV_CACHE_DIR=/workspace/.uvcache UV_PYTHON_INSTALL_DIR=/workspace/.uvpython
export XDG_CACHE_HOME=/workspace/.cache TMPDIR=/workspace/tmp UV_NO_PROGRESS=1
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$TMPDIR"
export PATH="/workspace/uvbin:$CARGO_HOME/bin:$PATH"
cd "$ALPA" || { echo "SETUP_FAIL: no alpasim dir"; exit 1; }

echo "STAGE_1_UV $(date -u +%H:%M:%S)"
if ! /workspace/uvbin/uv --version 2>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/workspace/uvbin INSTALLER_NO_MODIFY_PATH=1 sh || { echo SETUP_FAIL uv; exit 1; }
fi
uv --version || { echo SETUP_FAIL uv2; exit 1; }

echo "STAGE_2_RUST $(date -u +%H:%M:%S)"
if ! cargo --version 2>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal || { echo SETUP_FAIL rust; exit 1; }
fi
source "$CARGO_HOME/env" 2>/dev/null || true
cargo --version || { echo SETUP_FAIL cargo; exit 1; }

echo "STAGE_3_PYPROJECT $(date -u +%H:%M:%S)"
# back up original root pyproject once, install pared-down version (driver excluded)
[ -f pyproject.toml.orig ] || cp pyproject.toml pyproject.toml.orig
cp /workspace/pyproject_pared.toml pyproject.toml
echo "pyproject swapped (orig backed up to pyproject.toml.orig)"

echo "STAGE_4_SYNC $(date -u +%H:%M:%S)"
uv sync --extra core || { echo "SETUP_FAIL sync"; exit 1; }
echo "sync ok; venv at $ALPA/.venv"

echo "STAGE_5_PROTOS $(date -u +%H:%M:%S)"
( cd src/grpc && uv run compile-protos ) || { echo "SETUP_FAIL protos"; exit 1; }
ls src/grpc/alpasim_grpc/v0/*_pb2.py | head -3

echo "STAGE_6_UTILS_RS $(date -u +%H:%M:%S)"
uv pip install --force-reinstall -e src/utils_rs || { echo "SETUP_FAIL utils_rs"; exit 1; }

echo "STAGE_7_VERIFY $(date -u +%H:%M:%S)"
uv run python -c "import alpasim_grpc, alpasim_wizard, alpasim_runtime, alpasim_controller, alpasim_physics, alpasim_eval, alpasim_utils_rs; print('IMPORTS_OK')" 2>&1 | tail -5

echo "SETUP_DONE $(date -u +%H:%M:%S)"
