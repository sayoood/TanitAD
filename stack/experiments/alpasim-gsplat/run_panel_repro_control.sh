#!/usr/bin/env bash
# STREAM C — THE NEGATIVE CONTROL for the render A/B.
#
# The whole claim of the HQ panel is "these numbers moved BECAUSE the render changed".
# That attribution is only admissible if the loop is deterministic: if re-running the
# MORNING config reproduces the MORNING rollouts, then morning-vs-HQ is the render and
# nothing else. If it does not reproduce, the render delta is confounded with run-to-run
# noise and no morning-vs-HQ difference may be attributed to render quality at all.
#
# This matters more here than anywhere else in the programme: the renderer is a STEP
# FUNCTION of pose (discrete blend-order ties among 3.1 M gaussians) and a 0.1 px camera
# rotation has been measured to move the 2 s waypoint 6.65 m.
#
# Runs the morning config EXACTLY — no --cull-scale-quantile, no --sky-gain, no
# --all-dynamic-layers, i.e. every render flag at its default — into a separate dir.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=${1:-$HOME/cl_out_repro}
STARTS=${2:-0,17,34,51,68,85,102,119,136}
STEPS=${3:-50}
mkdir -p "$OUT"

declare -A CKPT=(
  [flagship-v1]=$HOME/models/flagship-v1-speedjerk/ckpt.pt
  [refc-base]=$HOME/models/refc-base/ckpt.pt
)

for ARM in flagship-v1 refc-base; do
  D="$OUT/panel_${ARM}_empty"
  echo "=== REPRO (morning config) $ARM / empty ==="
  python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
      --condition empty --starts "$STARTS" --steps "$STEPS" \
      --out "$D" || echo "FAILED repro $ARM"
done
echo "REPRO_CONTROL_DONE"
