#!/usr/bin/env bash
# THE SYNTHESIS ARM, CORRECTED: the ACCURACY lever + the SAFETY lever, WITHOUT the
# seed ladder.
#
# ⛔ WHY THE LADDER IS DROPPED, and it is a MEASUREMENT taken minutes before this
# arm was written (RESULT.md 7.9, `wk15_ladder`): a discrete rung set and a
# CONTINUOUS QUADRATIC PENALTY are ANTAGONISTIC. `wk15` alone realises 15 distinct
# curvatures spread over 0.0115-0.0530 because it can trade smoothly; add the rungs
# and the realised set collapses to TWO values — 0.0000 x18 and 0.0020 x22, i.e.
# 22 of 40 windows snap EXACTLY onto the SMALLEST rung — because a quadratic
# penalty always prefers the cheapest non-zero option available. Tactical lateral
# kappa falls 0.2611 -> 0.0000 and turn_right recall 0.5 -> 0.0: the arm stops
# making lateral decisions.
# A CONSTRAINT has no preference gradient, so the ladder helps it (`combined`
# reached kamm_over 0.0000). A PENALTY does, so the ladder hurts it. Running
# penalty + constraint + ladder would inherit the collapse and buy nothing.
#
# ⚠️ This is an ARM-SET amendment made on a measured mechanism BEFORE this arm
# produced any number — the same class as the wk1p5 -> l3ladder swap recorded in
# PREREG_COST_GEOMETRY.md 6. No criterion moves; both outcomes stay as registered.
# The mis-specified `best` (with the ladder) ran for ~1 minute and was killed by
# explicit PID; its dump is removed below so no partial panel can be mistaken for
# a result.
#
# ONE variable against `wk15` (the cap) and ONE against `kamm07` (the penalty).
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_margin/p4out"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CACHE=C:/Users/Admin/refav1_margin/p4/fp8
EPS=C:/Users/Admin/refav1_margin/p4/eps

# a SIBLING agent's `lonshift` arm holds the other slot; one arm = TWO processes
for i in $(seq 1 900); do
  n=$(powershell.exe -NoProfile -Command \
      "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*refav1' + '_arm.py*' }).Count" \
      2>/dev/null | tr -d '\r ')
  n=$(printf '%s' "${n:-9}" | tr -dc '0-9')
  [ $((i % 15)) -eq 0 ] && echo "ZZWAITJ-${i}-n${n}ZZ $(date -u +%FT%TZ)"
  [ "${n:-9}" -le 2 ] 2>/dev/null && { echo "ZZWAITJ-CLEAR-${i}ZZ $(date -u +%FT%TZ)"; break; }
  sleep 20
done

tag=best
rm -rf "$OUT/dump_$tag" "$OUT/rec_${tag}.json" "$OUT/${tag}.log"
echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
"$PY" "$M/taniteval/tools/refav1_arm.py" \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
  --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL" \
  --device cuda --episodes-n 0 --window-stride 16 --no-navshuf \
  --no-lead-block --cost-metric ccos \
  --cost-weights 0.0,15.11245,64.29715042415070 \
  --kamm-mu 0.7 \
  --plan-seed 0 \
  --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
  --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
echo "ZZQUEUEJ-DONE-$(date -u +%FT%TZ)ZZ"
