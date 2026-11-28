#!/bin/sh
# Wait for the dev-box GPU gate, then run G0 on the kit checkpoint.
# Assert on the ARTIFACT (JSON), never on $?. Python gets WINDOWS paths (an MSYS /c/ path inside a
# Python string is not converted -- that killed the first launch with FileNotFoundError).
BW='C:/Users/Admin/ev6_battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval"
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /c/Users/Admin/ev6_battery/code
$PY gpu_gate.py --wait --max-wait-s 43200 --out "$BW/raw/gate_g0_step1000.json" > /c/Users/Admin/ev6_battery/raw/gate_g0_step1000.log 2>&1
$PY -c "import json,sys; g=json.load(open(r'$BW/raw/gate_g0_step1000.json')); sys.exit(0 if g['ok'] else 3)" || { echo "ZZGATENOTPASSEDZZ" >> /c/Users/Admin/ev6_battery/raw/g0_step1000.log; exit 3; }
$PY reproduce_inrun_eval.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step1000.pt \
   --config D:/refcv6_eval_kit/ckpt/config.json \
   --metrics "$BW/raw/thor_reads/metrics_20260923T2313.jsonl" \
   --seeds 0,1,2,3,4,5,6,7 --mutation m1_no_equalize \
   --out "$BW/raw/g0_step1000.json" > /c/Users/Admin/ev6_battery/raw/g0_step1000.log 2>&1
[ -s /c/Users/Admin/ev6_battery/raw/g0_step1000.json ] && echo "ZZG0DONEZZ" >> /c/Users/Admin/ev6_battery/raw/g0_step1000.log || echo "ZZG0NOJSONZZ" >> /c/Users/Admin/ev6_battery/raw/g0_step1000.log
