#!/usr/bin/env bash
# exploratory (non-token) run of the gate's own record job on the shipped tip tree, arm $1
ARM=$1; R=/home/nvidia/r8ns/r9e1; T=$R/launch_tree; SHA=$(cat $R/launch_tree.commit); C12=${SHA:0:12}
ARGV=$T/stack/ops/runs.d/r9e1_${ARM}.argv.json
REC12="$(cat $R/gate3/REC_C12 2>/dev/null || echo d297ba563d8f)"; OV=$R/ovf_${ARM}_$REC12
# the record KIND follows the arm's argv (E1_OWN_RECORDS): a --r9-map-banded arm binds its /2-banded map record
MREC=$OV/run_map/g_map_overfit.json; MCLO=$OV/map_closure.json; BREC=$OV/run_box/gbo_binding.json; BCLO=$OV/box_closure.json
grep -q '"--r9-map-banded"' "$ARGV" 2>/dev/null && { MREC=$OV/run_map-banded/g_map_overfit.json; MCLO=$OV/map-banded_closure.json; }
PY=/home/nvidia/venvs/tanitad-train/bin/python
cd /tmp && PYTHONPATH=$T/stack:$T/stack/scripts:$T/taniteval OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= $PY $T/stack/scripts/launch_gate.py run \
  --profile refcv9-e1rig --checks G-MAP-OVERFIT,G-BOX-OVERFIT --tree $T --commit $SHA \
  --argv-file $T/stack/ops/runs.d/r9e1_${ARM}.argv.json --out-dir $R/gate3_explore/rec_${C12}_$ARM \
  --map-overfit-record $MREC --map-overfit-closure $MCLO \
  --box-overfit-record $BREC --box-overfit-closure $BCLO \
  > $R/gate3_explore_rec_${C12}_$ARM.log 2>&1
echo "rc=$?" >> $R/gate3_explore_rec_${C12}_$ARM.log
