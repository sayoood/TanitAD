#!/usr/bin/env bash
# Score every published closed-loop arm as a PAIRED MARGIN OVER `cl_ha0`.
# Same scorer (`cl_metrics.py`), same windows, same paired episode-cluster
# bootstrap as the panels being re-read. A>B means A is WORSE on an error metric.
set -u
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12
export OMP_NUM_THREADS=6
export PYTHONPATH=$HOME/tanitad_cl_v3/stack:$HOME/tanitad_cl_v3/stack/scripts:$HOME/tanitad_cl_v3/taniteval:$HOME/tanitad_cl_v3/taniteval/tools:$HOME/tanitad_cl_v3/stack/experiments/nurec-gsplat:$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat
cd "$HOME/tanitad_cl_v3/stack/experiments/alpasim-gsplat"
F=$HOME/cl_out_floor
V3=$HOME/cl_out_v3
HQ=$HOME/cl_out_hq
M=$F/metrics
mkdir -p "$M"
rm -f "$M/METRICS.DONE"
HA0=$F/panel_cl_ha0/rollouts_cl_ha0_empty.json

sc () {  # tag A B
  python cl_metrics.py --a "$2" --b "$3" --out "$M/$1.json" > /dev/null 2>"$M/$1.err" \
    && echo "OK__$1" || { echo "FAIL__$1"; tail -3 "$M/$1.err"; }
}

# --- today's v3-tree panel (2026-09-04) as margins over the floor -------------
sc V3_refcv3_vs_ha0      $V3/panel_refcv3_empty/rollouts_refcv3_empty.json           $HA0
sc V3_refcbase_vs_ha0    $V3/panel_refc-base_empty/rollouts_refc-base_empty.json     $HA0
sc V3_flagshipv1_vs_ha0  $V3/panel_flagship-v1_empty/rollouts_flagship-v1_empty.json $HA0
# --- the BANKED 2026-08-03 HQ panel as margins over the same floor ------------
sc HQ_flagshipv1_vs_ha0  $HQ/panel_flagship-v1_empty/rollouts_flagship-v1_empty.json $HA0
sc HQ_refcbase_vs_ha0    $HQ/panel_refc-base_empty/rollouts_refc-base_empty.json     $HA0
# --- the floor's own internal contrasts ---------------------------------------
sc FL_ha_vs_ha0          $F/panel_cl_ha/rollouts_cl_ha_empty.json                    $HA0
sc FL_haext_vs_ha0       $F/panel_cl_ha0_ext/rollouts_cl_ha0_ext_empty.json          $HA0
# --- CONTROLS: both must read delta 0.0 with a ZERO-WIDTH CI -------------------
sc CTRL_repro_ha0        $F/repro_cl_ha0/rollouts_cl_ha0_empty.json                  $HA0
sc CTRL_morn_ha0         $F/morn_cl_ha0/rollouts_cl_ha0_empty.json                   $HA0

python $HOME/floor_controls.py "$F" "$M/RAW_CONTROLS.json"
touch "$M/METRICS.DONE"
echo "FLOOR_METRICS_FINISHED"
