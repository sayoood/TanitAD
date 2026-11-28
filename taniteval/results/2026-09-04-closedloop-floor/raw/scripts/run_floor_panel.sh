#!/usr/bin/env bash
# CLOSED-LOOP TRIVIAL FLOOR PANEL (2026-09-04).
# Protocol IDENTICAL to run_panel_hq.sh and run_panel_v3.sh: same scene, same 9
# starts, same 50 ticks, same render flags, rolling shutter OFF, --condition empty.
# That identity is what makes the floor pairable with BOTH published panels.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl_v3/stack:$HOME/tanitad_cl_v3/stack/scripts:$HOME/tanitad_cl_v3/taniteval:$HOME/tanitad_cl_v3/taniteval/tools:$HOME/tanitad_cl_v3/stack/experiments/nurec-gsplat:$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat"
SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=$HOME/cl_out_floor
STARTS=0,17,34,51,68,85,102,119,136
STEPS=50
RQ="--cull-scale-quantile 0.95 --sky-gain 0.3"
mkdir -p "$OUT"
rm -f "$OUT/PANEL.DONE"

run () {   # tag arm extra...
  local TAG=$1 ARM=$2; shift 2
  local D="$OUT/${TAG}"
  echo "=== FLOOR $TAG (arm=$ARM starts $STARTS x $STEPS) $* ==="
  python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" \
      --condition empty --starts "$STARTS" --steps "$STEPS" --out "$D" "$@" \
      && echo "OK__${TAG}" || echo "FAIL__${TAG}"
}

run panel_cl_ha0      cl_ha0     $RQ
run panel_cl_ha       cl_ha      $RQ
run panel_cl_ha0_ext  cl_ha0_ext $RQ
# CONTROL 1 — determinism: the identical run again, must be bit-identical.
run repro_cl_ha0      cl_ha0     $RQ
# CONTROL 2 — render independence: the MORNING render (no cull, no sky). A policy
# that truly reads no pixels must produce a BIT-IDENTICAL trajectory.
run morn_cl_ha0       cl_ha0

touch "$OUT/PANEL.DONE"
echo "FLOOR_PANEL_FINISHED"
