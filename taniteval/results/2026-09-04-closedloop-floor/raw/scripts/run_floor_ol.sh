#!/usr/bin/env bash
# OPEN-LOOP sweep of the three TRIVIAL FLOOR arms on the SAME NuRec scene, with the
# SAME render config and the SAME 9 disjoint segments as the 2026-09-04 model sweep.
# One render pass drives all three, and the floor reads no pixels anyway.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl_v3/stack:$HOME/tanitad_cl_v3/stack/scripts:$HOME/tanitad_cl_v3/taniteval:$HOME/tanitad_cl_v3/taniteval/tools:$HOME/tanitad_cl_v3/stack/experiments/nurec-gsplat:$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat"
SCENE=$HOME/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
OUT=$HOME/cl_out_floor/openloop
rm -f $OUT/OL.DONE
mkdir -p "$OUT"
python openloop_drive.py --scene-dir "$SCENE" --out "$OUT" \
    --floor cl_ha0 --floor cl_ha --floor cl_ha0_ext \
    --layers background,road --cull-scale-quantile 0.95 --sky-gain 0.3 \
    --n-clusters 9 && echo "OK__OL_FLOOR" || echo "FAIL__OL_FLOOR"

M=$HOME/cl_out_floor/metrics
V3=$HOME/cl_out_v3/openloop
sc () { python cl_metrics.py --a "$2" --b "$3" --out "$M/$1.json" >/dev/null 2>"$M/$1.err" \
        && echo "OK__$1" || { echo "FAIL__$1"; tail -3 "$M/$1.err"; }; }
sc OL_refcv3_vs_ha0      $V3/rollouts_refcv3_openloop.json       $OUT/rollouts_cl_ha0_openloop.json
sc OL_refcbase_vs_ha0    $V3/rollouts_refc-base_openloop.json    $OUT/rollouts_cl_ha0_openloop.json
sc OL_flagshipv1_vs_ha0  $V3/rollouts_flagship-v1_openloop.json  $OUT/rollouts_cl_ha0_openloop.json
sc OL_ha_vs_ha0          $OUT/rollouts_cl_ha_openloop.json       $OUT/rollouts_cl_ha0_openloop.json
sc OL_haext_vs_ha0       $OUT/rollouts_cl_ha0_ext_openloop.json  $OUT/rollouts_cl_ha0_openloop.json
touch $OUT/OL.DONE
echo "FLOOR_OL_FINISHED"
