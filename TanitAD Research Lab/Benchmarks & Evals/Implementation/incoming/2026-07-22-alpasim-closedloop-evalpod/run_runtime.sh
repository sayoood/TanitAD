#!/bin/bash
# Run the AlpaSim runtime bare (no docker) for the localhost closed-loop rollout on tanitad-eval.
# Rewrites the wizard's container-mount paths (/mnt/nre-data) to real host paths first, since the
# wizard emits docker-compose-oriented paths. Assumes backing services already up
# (renderer :6011, physics :6006, controller :6007, driver :6789) and configs in /workspace/m2run.
set -uo pipefail
M2="${1:-/workspace/m2run}"
ALPA=/workspace/alpa-invest/alpasim
NRE_HOST="$ALPA/data/nre-artifacts"
sed -i "s#/mnt/nre-data#$NRE_HOST#g" "$M2/generated-user-config-0.yaml"
echo "scene data_dir now:"; grep -n data_dir "$M2/generated-user-config-0.yaml"
cd "$ALPA" || exit 1
rm -f /workspace/runtime.log
setsid .venv/bin/python -m alpasim_runtime.simulate \
  --user-config="$M2/generated-user-config-0.yaml" \
  --network-config="$M2/generated-network-config.yaml" \
  --log-dir="$M2" --log-level=INFO --array-job-dir="$M2" \
  --eval-config="$M2/eval-config.yaml" \
  </dev/null >/workspace/runtime.log 2>&1 &
echo "RUNTIME_PID=$!"
