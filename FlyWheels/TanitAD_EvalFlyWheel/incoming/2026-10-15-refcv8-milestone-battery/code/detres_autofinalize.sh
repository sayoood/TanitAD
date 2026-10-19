#!/bin/sh
# waits for the GPU chain (acq2 log marker), then applies the PREREG rule mechanically and, on R1 only, computes the amendment counts.  Assert on the json, not on exit codes.
P='C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery'
W='C:/Users/Admin/qland/work/detres'; PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe; export PYTHONIOENCODING=utf-8
n=0; until grep -qa "ZZDETRESRANZZ\|ZZDETRESNOLOCKZZ\|ZZDETRESNOCOMMITZZ" "$W/gpu_acq2.log" 2>/dev/null || [ $n -gt 3000 ]; do sleep 60; n=$((n+1)); done
$PY "$P/code/detres_finalize.py" --probe "$P/raw/step5000/detection_residual_probe.json" --g0 E:/refcv8_eval_kit/battery/step5000/g0.json --pack "$P/raw/step5000/detres/pack_analysis.json" >> "$W/autofinalize.log" 2>&1
V=$($PY -I -c "import json; print(json.load(open(r'$P/raw/step5000/detection_residual_probe.json')).get('DECISION',{}).get('verdict',''))")
if [ "$V" = "R1_NUMERICS_SUFFICIENT" ]; then
  $PY "$P/code/detres_amendment_count.py" --g0 E:/refcv8_eval_kit/battery/step5000/g0.json --probe "$P/raw/step5000/detection_residual_probe.json" --out "$P/raw/step5000/detres/amendment_count.json" >> "$W/autofinalize.log" 2>&1
fi
echo "ZZDETRESFINALIZEDZZ verdict=$V $(date +%FT%T)" >> "$W/autofinalize.log"
