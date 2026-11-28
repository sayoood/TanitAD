set -u
D=/workspace/data; ROOT=$D/refe_p5c; PY=/workspace/venv-teacher/bin/python; X=$ROOT/l5probe; PKG=$X/code_95b3d174
EV=$ROOT/evidence_sftt
export PYTHONPATH=$PKG/refe PYTHONUNBUFFERED=1 REFE_BACKBONE_ROOT=$D/backbones PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUPLAN_MAPS_ROOT=$D/nuplan-maps/nuplan-maps-v1.0 TANITAD_GATE_KEY_FILE=$ROOT/launch_gate.key
cd $PKG/refe || exit 9
ARGS="$(cat $ROOT/ARGS_SFTT.txt)"
env -u HF_TOKEN OMP_NUM_THREADS=4 $PY launch_gate.py run --profile refe_scorerft_t --pkg $PKG --args "$ARGS" --out-dir $X/lg --only G-LIVE --gate-bank $X/bank --gate-bank-n 64 --gate-lab-json $EV/lab.json --gate-mem-metrics $EV/mem_metrics.jsonl --gate-egr-json $EV/egr/g_egr_result.json > $X/l5probe.log 2>&1
echo "ZZEXIT $?" >> $X/l5probe.log
echo ZZL5PROBE_DONE >> $X/l5probe.log
