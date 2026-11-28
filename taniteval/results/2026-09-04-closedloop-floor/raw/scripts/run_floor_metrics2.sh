#!/usr/bin/env bash
# The models against the STRONGEST trivial floors (cl_ha0_ext = CTRA, cl_ha = held
# action), both tiers. `cl_ha0` is the bit-comparable bar; these two are the harder
# ones, and on this scene they SEPARATE from `cl_ha0` in open loop.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl_v3/stack:$HOME/tanitad_cl_v3/stack/scripts:$HOME/tanitad_cl_v3/taniteval:$HOME/tanitad_cl_v3/taniteval/tools:$HOME/tanitad_cl_v3/stack/experiments/nurec-gsplat:$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat"
F=$HOME/cl_out_floor; V3=$HOME/cl_out_v3; M=$F/metrics
rm -f $M/METRICS2.DONE
EXT=$F/panel_cl_ha0_ext/rollouts_cl_ha0_ext_empty.json
HA=$F/panel_cl_ha/rollouts_cl_ha_empty.json
OEXT=$F/openloop/rollouts_cl_ha0_ext_openloop.json
OHA=$F/openloop/rollouts_cl_ha_openloop.json
sc () { python cl_metrics.py --a "$2" --b "$3" --out "$M/$1.json" >/dev/null 2>"$M/$1.err" \
        && echo "OK__$1" || { echo "FAIL__$1"; tail -3 "$M/$1.err"; }; }
sc V3_refcv3_vs_haext     $V3/panel_refcv3_empty/rollouts_refcv3_empty.json           $EXT
sc V3_refcbase_vs_haext   $V3/panel_refc-base_empty/rollouts_refc-base_empty.json     $EXT
sc V3_flagshipv1_vs_haext $V3/panel_flagship-v1_empty/rollouts_flagship-v1_empty.json $EXT
sc V3_refcv3_vs_ha        $V3/panel_refcv3_empty/rollouts_refcv3_empty.json           $HA
sc V3_refcbase_vs_ha      $V3/panel_refc-base_empty/rollouts_refc-base_empty.json     $HA
sc OL_refcv3_vs_haext     $V3/openloop/rollouts_refcv3_openloop.json                  $OEXT
sc OL_refcbase_vs_haext   $V3/openloop/rollouts_refc-base_openloop.json               $OEXT
sc OL_flagshipv1_vs_haext $V3/openloop/rollouts_flagship-v1_openloop.json             $OEXT
sc OL_refcv3_vs_ha        $V3/openloop/rollouts_refcv3_openloop.json                  $OHA
sc OL_refcbase_vs_ha      $V3/openloop/rollouts_refc-base_openloop.json               $OHA
touch $M/METRICS2.DONE
echo "FLOOR_METRICS2_FINISHED"
