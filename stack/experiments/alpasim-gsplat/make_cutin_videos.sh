#!/usr/bin/env bash
# House-style closed-loop videos for the constructed close-following / cut-in conditions:
# camera + metric BEV + decoded tactical manoeuvre + strategic route + the CONSTRUCTED
# lead's headway/time-gap, with a COLLISION banner whenever headway <= 0.
# Each file is verified by DECODING it, never by exit code (overlay_video.py does that).
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
ROOT=${1:-$HOME/cutin_out}
VID=${2:-$HOME/cutin_videos}
mkdir -p "$VID"

for ARM in flagship-v1 refc-base; do
  for COND in lead8 cutin; do
    D="$ROOT/long_${ARM}_${COND}"
    R="$D/rollouts_${ARM}_${COND}.json"
    F="$D/frames_s0"
    [ -s "$R" ] || { echo "SKIP $ARM/$COND (no rollouts)"; continue; }
    [ -d "$F" ] || { echo "SKIP $ARM/$COND (no frames)"; continue; }
    echo "=== VIDEO $ARM / $COND ==="
    python overlay_video.py --rollouts "$R" --frames "$F" --scene-dir "$SCENE" \
        --out "$VID/${ARM}_${COND}.mp4" || echo "FAILED video $ARM $COND"
  done
done
ls -la "$VID"
echo "CUTIN_VIDEOS_DONE"
