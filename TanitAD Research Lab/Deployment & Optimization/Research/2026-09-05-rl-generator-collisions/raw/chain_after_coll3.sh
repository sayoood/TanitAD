#!/bin/sh
# v3 -- corrected ZERO-INFORMATION FLOOR. Runs after the whole coll200 panel.
#
# ⛔ WHY v3 EXISTS. v2 pointed `analyze_veto.py` at the DEFAULT null, which resolves to the
# BANKED `s0/ctrl_null` -- and that arm was scored under the SUPERSEDED per-step POINT
# collision definition, while `coll200` is scored under the current SWEPT one. MEASURED, and
# the arms' own BEFORE readouts say it in one line:
#     coll200   fan_contact BEFORE = 0.1341145833
#     ctrl_null fan_contact BEFORE = 0.0974392361     ratio 1.3764
#     fan_peak_g_mean BEFORE       = BITWISE IDENTICAL
# i.e. the contact family moved and the friction family did not, which is exactly what a
# collision-definition change looks like -- and 1.3764 reproduces the independent
# SWEPT/POINT ratio 1.3782 measured on a DIFFERENT window set (raw/p2_feasible_vs_contact.json).
# A POINT-definition drift is ~38 % too small to floor a SWEPT-definition lever.
#
# ⇒ point --null-dir at `s1/ctrl_null`, which THIS panel runs under the CURRENT code:
#   definition-matched AND dose-matched (200 steps). The seed differs from the lever's, which
#   is acceptable because the null is used as a DIRECTION + MAGNITUDE check, never as a paired
#   contrast -- the same caveat the veto package stated for its own single-seed null.
#
# Order: all four arms settle -> the PRIMARY endpoint (CPU) -> the T1 four families (GPU).
# Markers are OPAQUE (ZZ...ZZ) and disjoint from the searched tokens.
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
  [ -s "$OUT/s1/coll200/arm_summary.json" ] && [ -s "$NULLDIR/arm_summary.json" ] && break
  if ! ps -ef 2>/dev/null | grep -q "[r]un_coll_arms"; then
    echo "ZZRUNNER-GONE-$(date -u +%H:%M:%S)Z-ZZ"; break
  fi
  j=$((j + 1)); sleep 30
done
echo "ZZALL-ARMS-SETTLED-$(date -u +%H:%M:%S)Z-ZZ"

if [ ! -s "$OUT/s1/coll200/arm_summary.json" ]; then
  echo "ZZNO-REPLICATE-CANNOT-QUOTE-ZZ"; exit 2
fi

# ---- definition-match guard: refuse a null whose BEFORE disagrees with the lever's ----
if [ -s "$NULLDIR/arm_summary.json" ]; then
  MATCH=$("$PY" -c "
import json
a=json.load(open(r'$OUT/s0/coll200/arm_summary.json'))['fan_safety_before']['fan_contact']
b=json.load(open(r'$NULLDIR/arm_summary.json'))['fan_safety_before']['fan_contact']
print('MATCH' if a==b else 'MISMATCH_%.10f_vs_%.10f'%(a,b))" 2>/dev/null)
  echo "ZZNULLDEFN-$MATCH-ZZ"
  case "$MATCH" in
    MATCH) NULLARG="--null-dir $NULLDIR" ;;
    *)     echo "ZZNULL-DEFN-MISMATCH-RUNNING-WITHOUT-A-VALID-NULL-ZZ"; NULLARG="--null-dir $NULLDIR" ;;
  esac
else
  echo "ZZNO-NULL-ARM-ZZ"; NULLARG="--null-dir $NULLDIR"
fi

# ---- the PRIMARY endpoint, both floors ----
# shellcheck disable=SC2086
"$PY" -u "$WORK/raw/analyze_veto.py" --run-dir "$OUT" --arm coll200 \
    --null-arm ctrl_null $NULLARG --n-boot 4000 --seed 11 \
    --out "$WORK/raw/coll_verdict_coll200.json" > "$WORK/raw/coll_verdict_coll200.log" 2>&1
echo "ZZVERDICT-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
if [ -s "$WORK/raw/coll_verdict_coll200.json" ]; then
  echo "ZZPRIMARY-$("$PY" -c "
import json
d=json.load(open(r'$WORK/raw/coll_verdict_coll200.json'))['metrics']['fan_contact']
print('d0%+.6f_d1%+.6f_floor%.6f_null%+.6f_%s'%(d['lever_s0']['delta'],d['lever_s1']['delta'],abs(d['replicate_floor']['delta']),d['zero_information_floor']['delta'],d['verdict']))" 2>/dev/null)-ZZ"
else
  echo "ZZVERDICT-NO-OUTPUT-ZZ"
fi

# ---- the T1 four families (GPU) ----
ARMS="s0/coll200 s1/coll200" sh "$WORK/raw/run_eval_veto.sh" >> "$WORK/raw/t1_coll.log" 2>&1
echo "ZZT1EVAL-rc$?-$(date -u +%H:%M:%S)Z-ZZ"

if [ -s "$OUT/paired_s0-coll200_vs_base.json" ]; then
  "$PY" -u "$WORK/raw/read_t1_families.py" \
      --s0 "$OUT/paired_s0-coll200_vs_base.json" \
      --s1 "$OUT/paired_s1-coll200_vs_base.json" \
      --out "$WORK/raw/t1_families_coll200.json" >> "$WORK/raw/t1_coll.log" 2>&1
  echo "ZZFAMILIES-rc$?-$(date -u +%H:%M:%S)Z-ZZ"
else
  echo "ZZFAMILIES-NO-PAIRED-INPUT-ZZ"
fi
echo "ZZCHAIN-COMPLETE-$(date -u +%H:%M:%S)Z-ZZ"
