#!/usr/bin/env bash
# `best_seed1` — THE REPLICATE THE PACKAGE'S OWN VERDICT NEEDS.
#
# ⛔ WHY, and it is not a nicety. RESULT.md §7.11a replicated `combined`'s
# friction-circle result at --plan-seed 1 and the "zero" read 0.0741 with peak_g
# max 0.702 — OVER the mu = 0.7 circle. `best`'s kamm_over 0.0000 / peak_g
# 0.082 / 0.332 are UNREPLICATED at a single inference seed, and the one zero in
# that family that WAS replicated did not hold. Every lever verdict in this
# package rests on one inference seed plus a floor measured on one pair.
#
# ONE token differs from `best`: --plan-seed 0 -> 1. Nothing else. Verify against
# the LIVE process argv, not the intended command (§7.11a's own control).
#
# ⛔ BOTH OUTCOMES COMMITTED IN ADVANCE:
#   * kamm_over_rate reads 0.0000 again and the four families move by less than
#     the ccos_argmax/ccos_seed1 floor per metric => `best`'s safety result is
#     REPLICATED and may be quoted as a property rather than a single draw;
#   * any value > 0.0000, or a family metric moving more than its floor
#     => `best`'s zero is SEED-DEPENDENT exactly as `combined`'s was, the §7.10
#     claim is wrong as reported, and the correction is written in the same turn.
#
# ⚠️ Two SIBLING arms (`turnasym-ta_wk15_s0/s1`) hold the GPU; their slots are not
# mine. One arm = TWO python processes, so <= 2 means one arm is free.
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

for i in $(seq 1 1200); do
  n=$(powershell.exe -NoProfile -Command \
      "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*refav1' + '_arm.py*' }).Count" \
      2>/dev/null | tr -d '\r ')
  n=$(printf '%s' "${n:-9}" | tr -dc '0-9')
  [ $((i % 15)) -eq 0 ] && echo "ZZWAITK-${i}-n${n}ZZ $(date -u +%FT%TZ)"
  [ "${n:-9}" -le 2 ] 2>/dev/null && { echo "ZZWAITK-CLEAR-${i}ZZ $(date -u +%FT%TZ)"; break; }
  sleep 20
done

tag=best_seed1
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
  --plan-seed 1 \
  --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
  --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
echo "ZZQUEUEK-DONE-$(date -u +%FT%TZ)ZZ"
