#!/usr/bin/env bash
# Re-run exactly the panel cells that were produced by code later found to be wrong.
# Both defects were caught by controls, not by inspection, and both are recorded here so
# the re-run is auditable rather than a silent overwrite:
#
#   flagship-v1_empty  the counterfactual headway used a 2.0 m default half-length while
#                      every lead condition used the measured 1.542 m. The `behind`
#                      negative control surfaced it as a +0.458 m delta that cannot
#                      exist (behind and empty render and drive identically).
#   flagship-v1_cutin  the lateral ramp was anchored to the CLIP start, so 8 of the 9
#                      rollout clusters would have been plain lead vehicles labelled
#                      cut-ins. Now anchored to each rollout's first decision.
#
# Waits for the main panel to finish first so the two never contend for the GPU.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat:$HOME/alpasim/src/grpc
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=${1:-$HOME/cutin_out}
STARTS=0,15,30,45,60,75,90,105,120

until grep -q CUTIN_PANEL_DONE /tmp/cutin_panel.log; do sleep 20; done
echo "main panel done, re-running stale cells"

for SPEC in "flagship-v1 empty" "flagship-v1 cutin"; do
  set -- $SPEC; ARM=$1; COND=$2
  rm -rf "$OUT/${ARM}_${COND}"
  echo "=== RERUN $ARM / $COND ==="
  python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" \
      --ckpt "$HOME/models/flagship-v1-speedjerk/ckpt.pt" \
      --condition "$COND" --starts "$STARTS" --steps 50 \
      --out "$OUT/${ARM}_${COND}" || echo "FAILED rerun $ARM $COND"
done
rm -rf "$OUT/long_flagship-v1_cutin"
python closedloop_drive.py --scene-dir "$SCENE" --arm flagship-v1 \
    --ckpt "$HOME/models/flagship-v1-speedjerk/ckpt.pt" \
    --condition cutin --starts 0 --steps 150 \
    --out "$OUT/long_flagship-v1_cutin" --save-video-frames || echo "FAILED long rerun"
echo "CUTIN_RERUN_DONE"
