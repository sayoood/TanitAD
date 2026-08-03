#!/usr/bin/env bash
# STREAM C — score the IMPROVED-render panel: four families + ADE, paired CIs.
#
# Identical scorer, identical arguments to the morning run on this scene (--tracks only;
# no --renderable-from and no --lead-ref, matching results/metrics_*.json whose
# `renderable_restricted` and `lead_ref` are both null). Only the rollouts differ, and
# they differ only by the render.
#
# EIGHT CONTRASTS, every one paired on identical (start, k) windows:
#
#   A. THE PANEL, re-measured on the improved render
#      1. flagship-v1 vs refc-base within `empty`    — the morning HEADLINE contrast
#      2. flagship-v1 vs refc-base within `objects`
#      3. objects vs empty, flagship-v1              — does the arm react to traffic?
#      4. objects vs empty, refc-base
#
#   B. THE RENDER EFFECT ITSELF, per arm (HQ minus morning, same arm, same starts)
#      5. flagship-v1: HQ-empty vs morning-empty
#      6. refc-base:   HQ-empty vs morning-empty
#
#   C. THE NEGATIVE CONTROL / NOISE FLOOR (morning config re-run vs morning, same arm)
#      7. flagship-v1: repro-empty vs morning-empty
#      8. refc-base:   repro-empty vs morning-empty
#
#   D. THE MORNING ROLLOUTS, RE-SCORED WITH TODAY'S SCORER (9-12)
#      ⛔ REQUIRED, not optional. The published morning panel
#      (results/metrics_*.json) was produced BEFORE the route-head key fix, so it
#      records REF-C as exposing no strategic route logits — which is false. Comparing
#      published-morning against HQ would confound THE RENDER with A SCORER FIX. Every
#      render comparison in the report is HQ vs MORNING-RESCORED, same code, same day.
#
#   ⛔ (5,6) are ONLY interpretable against (7,8). If the loop is not deterministic, a
#   morning-vs-HQ difference is not attributable to the render. Run C before believing B.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
TRACKS=$SCENE/extracted/sequence_tracks.json
HQ=${1:-$HOME/cl_out_hq}
AM=${2:-$HOME/cl_out}          # the morning run
RP=${3:-$HOME/cl_out_repro}    # the morning config, re-run today
M="$HQ/metrics_hq"
mkdir -p "$M"

R () { echo "$1/panel_$2_$3/rollouts_$2_$3.json"; }
S () { python cl_metrics.py --a "$1" --b "$2" --tracks "$TRACKS" --out "$3" > /dev/null \
         && echo "scored $(basename $3)" || echo "FAILED $(basename $3)"; }

# --- A. the panel on the improved render ---------------------------------------
for COND in empty objects; do
  S "$(R $HQ flagship-v1 $COND)" "$(R $HQ refc-base $COND)" \
    "$M/HQ_flagship_vs_refc_${COND}.json"
done
for ARM in flagship-v1 refc-base; do
  S "$(R $HQ $ARM objects)" "$(R $HQ $ARM empty)" "$M/HQ_${ARM}_objects_vs_empty.json"
done

# --- B. the render effect, per arm ---------------------------------------------
for ARM in flagship-v1 refc-base; do
  S "$(R $HQ $ARM empty)" "$(R $AM $ARM empty)" "$M/RENDER_${ARM}_hq_vs_morning.json"
done

# --- C. the negative control: morning config re-run vs morning -----------------
for ARM in flagship-v1 refc-base; do
  if [ -s "$(R $RP $ARM empty)" ]; then
    S "$(R $RP $ARM empty)" "$(R $AM $ARM empty)" "$M/CONTROL_${ARM}_repro_vs_morning.json"
  else
    echo "SKIP control $ARM — no repro rollouts at $(R $RP $ARM empty)"
  fi
done

# --- D. the morning rollouts, re-scored with TODAY's scorer --------------------
for COND in empty objects; do
  S "$(R $AM flagship-v1 $COND)" "$(R $AM refc-base $COND)" \
    "$M/MORNRESC_flagship_vs_refc_${COND}.json"
done
for ARM in flagship-v1 refc-base; do
  S "$(R $AM $ARM objects)" "$(R $AM $ARM empty)" "$M/MORNRESC_${ARM}_objects_vs_empty.json"
done

ls -la "$M"
echo "SCORE_HQ_DONE"
