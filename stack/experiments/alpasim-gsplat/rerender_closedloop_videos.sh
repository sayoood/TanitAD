#!/usr/bin/env bash
# Re-draw the four CLOSED-LOOP videos from the ALREADY-BANKED HQ rollouts.
#
# ⛔ THIS DOES NOT RE-DRIVE ANYTHING. It re-runs `overlay_video.py` over
# `~/cl_out_hq/long_*/frames_s0` and the rollout JSON that was produced with the chosen
# render (all 4 layers + scale-cull 0.95 + gated sky 0.3). Every number in the frames and
# in the HUD is the SAME measurement as before — what changes is only the drawing:
#   1. a CLOSED-LOOP badge burned into the camera panel, so an open-loop clip and a
#      closed-loop clip can never be confused once the files are copied somewhere else;
#   2. a legend naming GROUND TRUTH / MODEL PREDICTION / DRIVEN, because two colours
#      without a key is not "distinguishable";
#   3. a thicker ground-truth polyline, which used to vanish under the plan exactly where
#      the arm was right — i.e. where the viewer most needs to see the two coincide.
#
# ⚠️ Re-driving would NOT be equivalent: the renderer is a step function of pose and a
# 0.1 px camera rotation has been measured to move the 2 s waypoint 6.65 m. Re-using the
# banked rollouts keeps these videos on the SAME numerical path as the published panel.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat" || exit 1

SCENE=${SCENE:-$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430}
OUT=${1:-$HOME/cl_out_hq}
VID=${2:-$HOME/cl_videos_hq2}
mkdir -p "$VID"

for ARM in flagship-v1 refc-base; do
  for COND in objects empty; do
    D="$OUT/long_${ARM}_${COND}"
    R="$D/rollouts_${ARM}_${COND}.json"
    F="$D/frames_s0"
    [ -s "$R" ] && [ -d "$F" ] || { echo "SKIP $ARM/$COND (no banked rollout)"; continue; }
    case "$COND" in
      objects) NAME="${ARM}_with_objects" ;;
      empty)   NAME="${ARM}_empty_road" ;;
    esac
    echo "=== CLOSED-LOOP VIDEO $ARM / $COND -> $NAME.mp4 ==="
    python overlay_video.py --rollouts "$R" --frames "$F" --scene-dir "$SCENE" \
        --tracks "$SCENE/extracted/sequence_tracks.json" --mode closed_loop \
        --out "$VID/${NAME}.mp4" || echo "FAILED video $ARM $COND"
  done
done
ls -la "$VID"
echo "CLOSEDLOOP_RERENDER_DONE"
