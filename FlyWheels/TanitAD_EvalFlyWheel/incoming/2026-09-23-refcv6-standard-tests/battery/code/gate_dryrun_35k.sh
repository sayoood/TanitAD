#!/bin/sh
# De-risking the FINAL's gate (SPEC A5): G0 + G0-A1 + G0-A2 on the POST-SWITCH step-35000 snapshot
# (the rolling ckpt.pt pulled read-only at step 35,000, md5 68a4ef3b... 3-way equal), on the 82c2331 tree
# with the run's post-switch config (md5 a3193a46...) and the md5-equal sidecar, against the run's own
# step-35,000 in-run eval row (it carries `eval_cascade` and corrected-clock labels). This is exactly the
# gate chain_final_v2.sh will run on the FINAL; a failure here is found ~a day before the headline needs it.
# Queue position (Master Mind order + this insert): step 30000 -> A6 -> THIS -> L3 -> 15k/20k -> FINAL.
# Markers in raw/step35000_gate/dryrun.log; l3_tag.sh waits for its end marker.
B=/c/Users/Admin/ev6_battery
BW='C:/Users/Admin/ev6_battery'
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
REL=FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
NEW='C:/Users/Admin/ev6_82c2331'
export REFCV6_REPO="$NEW" PYTHONPATH="$NEW/stack;$NEW/taniteval"
export OMP_NUM_THREADS=8 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
D=$B/raw/step35000_gate
DW=$BW/raw/step35000_gate
L=$D/dryrun.log
CK=D:/refcv6_eval_kit/ckpt/ckpt_35000.pt
CFG=$BW/raw/thor_reads/config_resume34500_20260926.json
MET=$BW/raw/thor_reads/metrics_20260926T1500.jsonl
mkdir -p $D
cd $B/code || exit 1
echo "ZZDRYSTARTZZ $(date +%FT%T)" >> $L
until grep -qE 'ZZA6(DONE|FAIL)ZZ' $B/raw/a6/a6_chain.log 2>/dev/null; do sleep 120; done
gw() { $PY -c "import sys; sys.path.insert(0, r'$BW/code'); import run_battery as RB; RB.gate_wait(r'$DW/$1')" > /dev/null 2>&1; }
gw gate_g0.json
$PY reproduce_inrun_eval.py --ckpt $CK --config $CFG --metrics $MET --seeds 0,1,2,3,4,5,6,7 \
    --mutation m1_no_equalize --out $DW/g0.json > $D/g0.log 2>&1
if [ -s $D/g0.json ]; then
  gw gate_g0_A1.json
  $PY g0_mutations.py --g0-json $DW/g0.json --out $DW/g0_A1.json > $D/g0_A1.log 2>&1
  gw gate_g0_A2.json
  $PY wrapper_probe.py --ckpt $CK --config $CFG --out $DW/g0_A2_wrapper_probe.json \
      --g0-a1-json $DW/g0_A1.json > $D/g0_A2_wrapper_probe.log 2>&1
  [ -s $D/g0_A2_wrapper_probe.json ] && $PY a2_table.py "$DW/g0_A2_wrapper_probe.json" > $D/g0_A2_table.md 2>&1
  [ -s $D/g0.json ] && $PY g0_table.py "$DW/g0.json" > $D/g0_table.md 2>&1
  v=$($PY -c "import json; r=json.load(open(r'$DW/g0_A2_wrapper_probe.json')) if __import__('os').path.exists(r'$DW/g0_A2_wrapper_probe.json') else {}; g=json.load(open(r'$DW/g0.json')); print('G0', (g.get('verdict') or {}).get('G0'), 'G0-A2', r.get('G0_A2'))" 2>&1 | tail -1)
  echo "ZZDRYDONEZZ $v $(date +%FT%T)" >> $L
else
  echo "ZZDRYFAILZZ no g0.json $(date +%FT%T)" >> $L
fi
$PY bank_tag.py "$DW" "$PKG/raw/step35000_gate" > $D/bank.log 2>&1
{ echo ""; echo "## $(date +%F) FINAL-gate dry run: G0/A1/A2 on the post-switch step-35000 snapshot, 82c2331 tree (gate_dryrun_35k.sh)"
  for f in $(ls "$PKG/raw/step35000_gate"); do [ -f "$PKG/raw/step35000_gate/$f" ] && echo "$REL/raw/step35000_gate/$f"; done
} >> "$PKG/LANDING_READY.txt"
