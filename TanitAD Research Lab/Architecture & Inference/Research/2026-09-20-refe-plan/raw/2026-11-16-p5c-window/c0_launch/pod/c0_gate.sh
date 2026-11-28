#!/bin/bash
D=/workspace/data
ROOT=$D/refe_p5c
PKG=$ROOT/code_33824420
PY=/workspace/venv-teacher/bin/python
export PYTHONPATH=$PKG/refe PYTHONUNBUFFERED=1
export REFE_BACKBONE_ROOT=$D/backbones PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUPLAN_MAPS_ROOT=$D/nuplan-maps/nuplan-maps-v1.0
export TANITAD_GATE_KEY_FILE=$ROOT/launch_gate.key
cd $PKG/refe || exit 9
ARGS="$(cat $ROOT/ARGS_C0.txt)"
EV=$ROOT/evidence_c0
rm -rf $ROOT/lg_C0 $ROOT/gate_bank_C0; mkdir -p $ROOT/lg_C0
env -u HF_TOKEN OMP_NUM_THREADS=4 $PY launch_gate.py run --profile refe_p5c_c0 --pkg $PKG --args "$ARGS" --out-dir $ROOT/lg_C0 --gate-bank $ROOT/gate_bank_C0 --gate-bank-n 64 --gate-lab-json $EV/lab.json --gate-mem-metrics $EV/mem_metrics.jsonl --gate-egr-json $EV/egr/g_egr_result.json > $ROOT/launch_gate_C0.log 2>&1
echo "ZZEXIT $?" >> $ROOT/launch_gate_C0.log
tok="$(ls $ROOT/lg_C0/PASS_*.json 2>/dev/null | head -n 1)"
if [ -n "$tok" ]; then
  env -u HF_TOKEN $PY launch_gate.py verify --profile refe_p5c_c0 --token "$tok" --args "$ARGS" --pkg $PKG --json $ROOT/token_verify_C0.json > $ROOT/token_verify_C0.log 2>&1
  echo "VERIFY_RC $?" >> $ROOT/token_verify_C0.log
  env -u HF_TOKEN $PY $PKG/refe/train.py $ARGS --preflight > $ROOT/preflight_C0.log 2>&1
  echo "PREFLIGHT_RC $?" >> $ROOT/preflight_C0.log
fi
echo "ZZGATE_SCRIPT_DONE" >> $ROOT/launch_gate_C0.log
