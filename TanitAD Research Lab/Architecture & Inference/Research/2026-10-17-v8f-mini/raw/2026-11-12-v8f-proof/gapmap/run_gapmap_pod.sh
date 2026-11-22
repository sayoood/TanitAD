#!/bin/bash
set -u
GM=/workspace/v8proof/gapmap
stage="${1:-}"
case "$stage" in
  controls|names) ;;
  *) echo "usage: run_gapmap_pod.sh controls|names"; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
mkdir -p "$GM/read"
/workspace/venv/bin/python "$GM/kit/gapmap_t0.py" --stage "$stage" \
  --kit-gap /workspace/v8proof/gaperr/kit_gap \
  --join /workspace/v8proof/hierrep/pins/join_eval141.npz \
  --base /workspace/v8proof/cropscale/gapfiles/VGAPLE.npz \
  --gap-record /workspace/v8proof/gaperr/gapfiles/GAPERR_GAPFILES_RECORD.json \
  --tests-out "$GM/lead_tests/lead_pytest_gapmap.out" \
  --out "$GM/read/GAPMAP_T0.json" 2>&1 | tee "$GM/read/$stage.out"
rc=${PIPESTATUS[0]}
echo "ZZ RUN-GAPMAP $stage DONE rc $rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$GM/read/$stage.out"
exit "$rc"
