#!/usr/bin/env bash
# exploratory (non-token) run of the gate's own record job on the shipped tip tree, arm $1
ARM=$1; R=/home/nvidia/r8ns/r9e1; T=$R/launch_tree; SHA=$(cat $R/launch_tree.commit); C12=${SHA:0:12}
OV=$R/ovf_${ARM}_d297ba563d8f
PY=/home/nvidia/venvs/tanitad-train/bin/python
cd /tmp && PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= $PY $T/stack/scripts/launch_gate.py run \
  --profile refcv9-e1rig --checks G-MAP-OVERFIT,G-BOX-OVERFIT --tree $T --commit $SHA \
  --argv-file $T/stack/ops/runs.d/r9e1_${ARM}.argv.json --out-dir $R/gate3_explore/rec_${C12}_$ARM \
  --map-overfit-record $OV/run_map/g_map_overfit.json --map-overfit-closure $OV/map_closure.json \
  --box-overfit-record $OV/run_box/gbo_binding.json --box-overfit-closure $OV/box_closure.json \
  > $R/gate3_explore_rec_${C12}_$ARM.log 2>&1
echo "rc=$?" >> $R/gate3_explore_rec_${C12}_$ARM.log
