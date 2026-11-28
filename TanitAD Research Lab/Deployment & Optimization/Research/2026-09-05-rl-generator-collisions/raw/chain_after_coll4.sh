#!/bin/sh
# v4 -- the T1 four-family evals are DROPPED, and this file records why.
#
# The Master Mind's escalation (24bd5e8) + M39 (c9ab82c) retire `fan_contact` as the RL PRIMARY
# endpoint: a contact-stage projection takes it to a STRUCTURAL zero at +0.0000 m ADE and zero
# GPU. Spending ~40 GPU-minutes on two T1 rollouts to price an arm against a RETIRED endpoint is
# precisely the spend the escalation objects to, so it does not happen.
#
# What DOES happen: the arms in flight finish (the escalation explicitly asks for this -- they
# establish the rig's noise floor under H-ESTIM-SEED-1) and the CPU-only two-floor verdict is
# computed and banked, because it is the noise-floor artifact and it costs no GPU.
set -u
PY="/c/Users/Admin/venvs/tanitad/Scripts/python.exe"
REPO="/c/Users/Admin/refcv4b_repo"
OUT="/c/Users/Admin/veto_run/run"
WORK="/c/Users/Admin/tanitad-rlgen"
NULLDIR="$OUT/s1/ctrl_null"
export TANITAD_REPO="$REPO" PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
export PYTHONPATH="$REPO/stack;$REPO"

j=0
while [ $j -lt 480 ]; do
  [ -s "$OUT/s0/ctrl0/arm_summary.json" ] && [ -s "$NULLDIR/arm_summary.json" ] && break
  if ! ps -ef 2>/dev/null | grep -q "[r]un_coll_arms"; then
    echo "ZZRUNNER-GONE-$(date -u +%H:%M:%S)Z-ZZ"; break
  fi
  j=$((j + 1)); sleep 30
done
echo "ZZALL-ARMS-SETTLED-$(date -u +%H:%M:%S)Z-ZZ"

if [ -s "$NULLDIR/arm_summary.json" ]; then
  MATCH=$("$PY" -c "
import json
a=json.load(open(r'$OUT/s0/coll200/arm_summary.json'))['fan_safety_before']['fan_contact']
b=json.load(open(r'$NULLDIR/arm_summary.json'))['fan_safety_before']['fan_contact']
print('MATCH' if a==b else 'MISMATCH_%.10f_vs_%.10f'%(a,b))" 2>/dev/null)
  echo "ZZNULLDEFN-$MATCH-ZZ"
else
  echo "ZZNO-NULL-ARM-ZZ"
fi

"$PY" -u "$WORK/raw/analyze_veto.py" --run-dir "$OUT" --arm coll200 \
    --null-arm ctrl_null --null-dir "$NULLDIR" --n-boot 4000 --seed 11 \
    --out "$WORK/raw/coll_verdict_coll200.json" > "$WORK/raw/coll_verdict_coll200.log" 2>&1
echo "ZZVERDICT-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
if [ -s "$WORK/raw/coll_verdict_coll200.json" ]; then
  echo "ZZPRIMARY-$("$PY" -c "
import json
d=json.load(open(r'$WORK/raw/coll_verdict_coll200.json'))['metrics']['fan_contact']
print('d0%+.6f_d1%+.6f_floor%.6f_null%+.6f_%s'%(d['lever_s0']['delta'],d['lever_s1']['delta'],abs(d['replicate_floor']['delta']),d['zero_information_floor']['delta'],d['verdict']))" 2>/dev/null)-ZZ"
fi
echo "ZZT1-DROPPED-BY-ESCALATION-24bd5e8-ZZ"
echo "ZZCHAIN-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"
