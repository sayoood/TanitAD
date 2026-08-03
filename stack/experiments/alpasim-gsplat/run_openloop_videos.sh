#!/usr/bin/env bash
# OPEN-LOOP panel + videos on the Jetson Thor, with the 2026-08-03 CHOSEN render config.
#
# ⭐ OPEN LOOP: the ego follows the LOGGED trajectory. Each frame is rendered at the pose
# the rig actually had, the model consumes it and emits a plan, and the plan is scored
# against the log's own future motion. THE MODEL NEVER DRIVES. This is the experiment the
# programme had never run: every AlpaSim number so far is closed loop, where perception
# error and control drift are confounded (MEASURED 2026-08-03: flagship v1's driven path
# moved a mean 9.05 m from a RENDER CHANGE ALONE).
#
# RENDER (unchanged from run_quality_videos.sh, deliberately, so open and closed loop are
# on the same pixels): all 4 layers + scale-cull 0.95 + gated sky 0.3.
#   grad-NCC 0.2774 -> 0.3424 (+23.4 %), negative control 5/5, ~36 ms/frame.
#   run_dir = ~/rq_out/panel6_chosen.
# ⛔ Rolling shutter is measured-better (0.3747) at 161x the cost and is OFF. ROLL=1 turns
# it on; if you do, SAY SO and quote the cost.
#
# ⭐ BOTH ARMS RUN IN ONE PROCESS OFF ONE RENDER PASS, so they are paired on IDENTICAL
# pixels by construction rather than by a determinism argument. The per-tick frame md5 is
# recorded in every rollout file.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12:${CPATH:-}
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat" || exit 1

SCENE=${SCENE:-$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430}
OUT=${1:-$HOME/ol_out}
VID=${2:-$HOME/ol_videos}
CLUSTERS=${CLUSTERS:-9}
ROLL=${ROLL:-0}
mkdir -p "$OUT" "$VID"

RQ="--cull-scale-quantile 0.95 --sky-gain 0.3"
[ "$ROLL" = "1" ] && RQ="$RQ --rolling-shutter"
echo "RENDER-QUALITY FLAGS: $RQ   (+ --all-dynamic-layers for the objects condition)"

CK_FLAG=$HOME/models/flagship-v1-speedjerk/ckpt.pt
CK_REFC=$HOME/models/refc-base/ckpt.pt

# ---- 1. the two open-loop sweeps ------------------------------------------------- #
# `objects` renders the scene's own dynamic layers (the physically correct scene);
# `empty` is the matched control on the SAME logged poses, so an objects-vs-empty
# contrast in open loop cannot be explained by control drift the way the closed-loop
# one can.
for COND in objects empty; do
  EXTRA="$RQ"
  [ "$COND" = "objects" ] && EXTRA="$RQ --all-dynamic-layers"
  D="$OUT/$COND"
  echo "=== OPEN-LOOP SWEEP: $COND ==="
  python openloop_drive.py --scene-dir "$SCENE" \
      --ckpt flagship-v1=$CK_FLAG --ckpt refc-base=$CK_REFC \
      --out "$D" --n-clusters "$CLUSTERS" --save-video-frames $EXTRA \
      || { echo "FAILED sweep $COND"; continue; }
done

# ---- 2. score: paired, four families, same windows -------------------------------- #
for COND in objects empty; do
  D="$OUT/$COND"
  A="$D/rollouts_flagship-v1_openloop.json"
  B="$D/rollouts_refc-base_openloop.json"
  [ -s "$A" ] && [ -s "$B" ] || { echo "SKIP score $COND"; continue; }
  TR="$SCENE/extracted/sequence_tracks.json"
  REND=""
  [ -s "$D/renderable.json" ] && REND="--renderable-from $D/renderable.json"
  echo "=== SCORE $COND ==="
  python cl_metrics.py --a "$A" --b "$B" --tracks "$TR" $REND \
      --out "$OUT/OL_flagship_vs_refc_$COND.json" > /dev/null \
      || { echo "FAILED score $COND"; continue; }
  python ol_report.py --panel "$OUT/OL_flagship_vs_refc_$COND.json" \
      --out "$OUT/OL_PANEL_$COND.md" --json-out "$OUT/OL_AUDIT_$COND.json" \
      --title "OPEN-LOOP panel — flagship v1 vs REF-C base, $COND"
done

# ---- 3. videos -------------------------------------------------------------------- #
for COND in objects empty; do
  D="$OUT/$COND"
  [ -d "$D/frames" ] || { echo "SKIP video $COND (no frames)"; continue; }
  for ARM in flagship-v1 refc-base; do
    R="$D/video_${ARM}_openloop.json"
    [ -s "$R" ] || { echo "SKIP video $ARM/$COND"; continue; }
    case "$COND" in
      objects) NAME="${ARM}_openloop_with_objects" ;;
      empty)   NAME="${ARM}_openloop_empty_road" ;;
    esac
    echo "=== VIDEO $ARM / $COND -> $NAME.mp4 ==="
    python overlay_video.py --rollouts "$R" --frames "$D/frames" \
        --scene-dir "$SCENE" --tracks "$SCENE/extracted/sequence_tracks.json" \
        --mode open_loop --out "$VID/${NAME}.mp4" || echo "FAILED video $ARM $COND"
  done
done
ls -la "$VID"
echo "OPENLOOP_DONE"
