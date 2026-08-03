#!/usr/bin/env bash
# STREAM B: the close-following / cut-in discriminating panel.
#
# The scene has NO close-following geometry (two probes, scene_geometry.py), so the lead
# conditions are CONSTRUCTED by synth_actor.py from the scene's own gaussians. Read that
# module's header before quoting anything from here.
#
# CONDITIONS, and why each one is in the panel:
#   empty   the control — nothing drawn.
#   behind  the NEGATIVE CONTROL — the same gaussians posed behind the camera. Its delta
#           vs `empty` is the harness NOISE FLOOR; no smaller effect is interpretable.
#   lead25/lead15/lead8   the DOSE-RESPONSE. A real reaction is monotone in headway.
#   cutin   the named follow-up: a lateral entry into the ego lane at ~1 s of gap.
#
# Starts are 0..120 (9 clusters) x 50 steps so that start+warm+steps+lead_dt stays inside
# the 200-tick clip and the lead is never an extrapolated pose.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat:$HOME/alpasim/src/grpc
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=${1:-$HOME/cutin_out}
STARTS=${2:-0,15,30,45,60,75,90,105,120}
STEPS=${3:-50}
mkdir -p "$OUT"

declare -A CKPT=(
  [flagship-v1]=$HOME/models/flagship-v1-speedjerk/ckpt.pt
  [refc-base]=$HOME/models/refc-base/ckpt.pt
)

for ARM in flagship-v1 refc-base; do
  for COND in empty behind lead25 lead15 lead8 cutin; do
    D="$OUT/${ARM}_${COND}"
    if [ -s "$D/rollouts_${ARM}_${COND}.json" ]; then
      echo "=== SKIP $ARM/$COND (already present) ==="; continue
    fi
    echo "=== $ARM / $COND  starts=$STARTS steps=$STEPS ==="
    python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
        --condition "$COND" --starts "$STARTS" --steps "$STEPS" \
        --out "$D" || echo "FAILED $ARM $COND"
  done
done

# one long rollout per arm at the closest lead, WITH frames, for the video deliverable
for ARM in flagship-v1 refc-base; do
  for COND in lead8 cutin; do
    D="$OUT/long_${ARM}_${COND}"
    [ -s "$D/rollouts_${ARM}_${COND}.json" ] && continue
    python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
        --condition "$COND" --starts 0 --steps 150 --out "$D" --save-video-frames \
        || echo "FAILED long $ARM $COND"
  done
done
echo "CUTIN_PANEL_DONE"
