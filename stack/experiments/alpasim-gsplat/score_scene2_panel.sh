#!/usr/bin/env bash
# Score the scene-2 REAL close-following panel: four families + ADE, paired CIs.
#
# THREE contrasts, all paired on identical (start, k) windows:
#   1. objects vs empty, PER ARM      — does the arm react to REAL close traffic?
#      (`empty` is the matched control: same starts, same steps, nothing dynamic drawn.)
#   2. flagship-v1 vs refc-base, within `objects` — which arm handles it better?
#   3. flagship-v1 vs refc-base, within `empty`   — the no-traffic reference for (2),
#      so a cross-arm difference that exists WITHOUT traffic is not attributed to it.
#
# --renderable-from restricts the REAL-lead search to the 92 tracks the renderer actually
# draws, so distance-keeping is never credited against an agent the model cannot see.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat:$HOME/alpasim/src/grpc
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

OUT=${1:-$HOME/scene2_out}
SCENE=$HOME/nurec_scenes/sample_set/26.04_release/7c72937c-c620-4776-9555-d57222c0081f
TRACKS=$SCENE/extracted/sequence_tracks.json
AMAP=$HOME/cutin_out/scene2/actor_map_7c72937c.json
mkdir -p "$OUT/metrics"

R () { echo "$OUT/$1/rollouts_$2_$3.json"; }

# 1. objects vs empty, per arm
for ARM in flagship-v1 refc-base; do
  python cl_metrics.py \
      --a "$(R ${ARM}_objects $ARM objects)" \
      --b "$(R ${ARM}_empty  $ARM empty)" \
      --tracks "$TRACKS" --renderable-from "$AMAP" \
      --out "$OUT/metrics/${ARM}_objects_vs_empty.json" > /dev/null || echo "FAILED $ARM"
  echo "scored $ARM objects-vs-empty"
done

# 2/3. cross-arm within each condition
for COND in objects empty; do
  python cl_metrics.py \
      --a "$(R flagship-v1_$COND flagship-v1 $COND)" \
      --b "$(R refc-base_$COND  refc-base  $COND)" \
      --tracks "$TRACKS" --renderable-from "$AMAP" \
      --out "$OUT/metrics/flagship_vs_refc_${COND}.json" > /dev/null || echo "FAILED $COND"
  echo "scored flagship-vs-refc $COND"
done
echo "SCENE2_SCORING_DONE"
