#!/usr/bin/env bash
# STREAM D — CUT-IN TARGETED PANEL, scene 7c72937c.
#
# Rollout starts are placed so the RENDERABLE cut-in entry (wall tick 118, track 117 /
# id 30) lands at a KNOWN, SPREAD-OUT step of every rollout. First decision tick is
# start + 9 (NEED_FRAMES-1 warm frames on the logged path), so
#       k_entry = 118 - (start + 9)
# start   75  78  81  84  87  90  93  96  99 102 105 109
# k_entry 34  31  28  25  22  19  16  13  10   7   4   0
# 12 clusters of genuine exposure, against the predecessor's 3.
#
# Render config: ALL DEFAULTS (sky-gain 0, no scale cull, dynamic_rigids only) so this
# panel is directly comparable to the banked scene2 panel it re-powers. The +23.4 %
# quality config is opt-in and is NOT used here.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl/stack:$HOME/tanitad_cl/stack/scripts:$HOME/tanitad_cl/taniteval:$HOME/tanitad_cl/stack/experiments/nurec-gsplat:$HOME/tanitad_cl/stack/experiments/alpasim-gsplat:$HOME/alpasim/src/grpc
cd "$HOME/tanitad_cl/stack/experiments/alpasim-gsplat"

SCENE=$HOME/nurec_scenes/sample_set/26.04_release/7c72937c-c620-4776-9555-d57222c0081f
OUT=$HOME/cutin_targeted_out
STARTS=75,78,81,84,87,90,93,96,99,102,105,109
STEPS=50
mkdir -p "$OUT"

# provenance: the exact renderer this panel ran against
md5sum gsplat_renderer.py closedloop_drive.py cl_metrics.py actor_map.py \
       scene_geometry.py > "$OUT/CODE_MD5.txt" 2>&1
python -c "import gsplat,torch;print('gsplat',gsplat.__version__,'torch',torch.__version__)" \
       >> "$OUT/CODE_MD5.txt" 2>&1

declare -A CKPT=(
  [flagship-v1]=$HOME/models/flagship-v1-speedjerk/ckpt.pt
  [refc-base]=$HOME/models/refc-base/ckpt.pt
)

for ARM in flagship-v1 refc-base; do
  for COND in empty objects; do
    D="$OUT/${ARM}_${COND}"
    if [ -s "$D/rollouts_${ARM}_${COND}.json" ]; then
      echo "=== SKIP $ARM/$COND ==="; continue
    fi
    echo "=== $ARM / $COND  starts=$STARTS steps=$STEPS ==="
    python closedloop_drive.py --scene-dir "$SCENE" --arm "$ARM" --ckpt "${CKPT[$ARM]}" \
        --condition "$COND" --starts "$STARTS" --steps "$STEPS" --out "$D" \
        || echo "FAILED $ARM $COND"
  done
done

# NEGATIVE CONTROL — DETERMINISM. Same arm, same condition, same starts, run again.
# The only true instrument null available in this design: any non-zero delta here is a
# bug, not behaviour.
D="$OUT/flagship-v1_objects_REPEAT"
if [ ! -s "$D/rollouts_flagship-v1_objects.json" ]; then
  echo "=== DETERMINISM CONTROL: flagship-v1 / objects (repeat) ==="
  python closedloop_drive.py --scene-dir "$SCENE" --arm flagship-v1 \
      --ckpt "${CKPT[flagship-v1]}" --condition objects --starts "$STARTS" \
      --steps "$STEPS" --out "$D" || echo "FAILED repeat"
fi
echo "CUTIN_TARGETED_PANEL_DONE"
