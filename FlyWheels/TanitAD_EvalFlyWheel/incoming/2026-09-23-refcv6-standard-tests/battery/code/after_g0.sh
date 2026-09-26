#!/bin/sh
# Wait for the step-1000 G0 artifact, re-judge it with the corrected COUNT rule (zero GPU), then
# run the unattended chain. Markers are opaque.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" PYTHONIOENCODING=utf-8
until grep -qE 'ZZG0DONEZZ|ZZG0NOJSONZZ|ZZGATENOTPASSEDZZ' $B/raw/g0_step1000.log 2>/dev/null; do sleep 20; done
if grep -q 'ZZG0DONEZZ' $B/raw/g0_step1000.log; then
  cp $B/raw/g0_step1000.json $B/raw/g0_step1000.as_run.json
  cd $B/code && $PY rejudge_g0.py "$BW/raw/g0_step1000.json" > $B/raw/g0_step1000.rejudge.log 2>&1
  G=$($PY -c "import json; print(json.load(open(r'$BW/raw/g0_step1000.json',encoding='utf-8'))['verdict']['G0'])")
  echo "ZZG0VERDICT_${G}ZZ $(date +%FT%T)" >> $B/raw/chain.log
  if [ "$G" = "PASS" ]; then sh $B/code/chain_after_g0.sh; else echo "ZZCHAINSTOPPEDZZ G0=$G" >> $B/raw/chain.log; fi
else
  echo "ZZCHAINSTOPPEDZZ no G0 artifact" >> $B/raw/chain.log
fi
