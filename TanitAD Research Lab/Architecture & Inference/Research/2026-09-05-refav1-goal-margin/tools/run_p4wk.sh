#!/usr/bin/env bash
# P4-WK — IS `W_KAPPA` THE BINDING LATERAL TERM? The SHIPPED cost weights.
#
# ⭐ WHY, and it is the arm that P4 arm A made obvious. MEASURED (P4_RESULT.md,
# 40 windows, commit abded60): with `--cost-weights 0.0,0.0,64.297` — W_KAPPA,
# the CURVATURE PENALTY, exactly ZERO — refav1 emits curvature on 91 % of
# GT-turn windows AND 90 % of GT-STRAIGHT windows, both SATURATING the
# `kappa_max = 0.2` clip, with direction correct on 45 % of turns, and the
# planner scores ADE 1.8944 against 0.8772 for simply holding the measured
# (a0, kappa0). It CHOSE those plans (baseline_won_frac 0.075). The search is
# not failing to find what the cost wants; the cost wants the wrong thing.
#
# This arm changes EXACTLY ONE THING against arm A: the weights, from the zeroed
# triple to the SHIPPED {0.02, 0.05, 0.1}. Same metric (cos), same plan seed,
# same episodes, same stride — so it is paired with arm A window-for-window and
# isolates the curvature penalty.
#
# ⛔ BOTH OUTCOMES COMMITTED IN ADVANCE (banked in P4_RESULT.md, commit abded60,
# BEFORE this arm ran):
#   * the straight-window curvature rate falls well below 0.90 while the turn
#     rate stays high => the curvature penalty IS the binding lateral term, the
#     "the weights are inert" claim does not hold under `cos`, and every arm on
#     the zeroed triple needs re-reading;
#   * it does not move => the weights really are inert here, the thrashing is
#     the goal term or the search itself, and the vocabulary work becomes the
#     live lever again. A real outcome, not a failed experiment.
#
# ⛔ Waits for ABSENCE of sibling arms, never for a success marker: a wait keyed
# on success polls forever when the thing it waits for crashes.
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_margin/p4out"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"

for i in $(seq 1 1080); do
  live=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -like '*refav1_arm.py*' }).Count" 2>/dev/null | tr -d '\r ')
  # NO '|| echo 0': grep -c prints 0 AND exits 1, so $(grep -c ... || echo 0)
  # captures TWO lines of 0 and passes a string test against 0.
  dm=$(grep -c "P4C_DONE" "$OUT/driverC.log" 2>/dev/null)
  dm=$(printf '%s' "${dm:-0}" | tr -dc '0-9')
  live=$(printf '%s' "${live:-1}" | tr -dc '0-9')
  echo "[waitWK $i] refav1_arm procs=${live:-?} p4c_done=${dm:-?} $(date -u +%FT%TZ)"
  if [ "${dm:-0}" -ge 1 ] 2>/dev/null && [ "${live:-1}" -eq 0 ] 2>/dev/null; then break; fi
  sleep 20
done

LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CACHE=C:/Users/Admin/refav1_margin/p4/fp8
EPS=C:/Users/Admin/refav1_margin/p4/eps

"$PY" "$M/taniteval/tools/refav1_arm.py" \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
  --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL" \
  --device cuda --episodes-n 0 --window-stride 16 --no-navshuf \
  --no-lead-block --cost-metric cos \
  --cost-weights 0.02,0.05,0.1 \
  --plan-seed 0 \
  --dump-dir "$OUT/dump_cos_wk" --out "$OUT/rec_cos_wk.json" \
  --arm "refav1-21109-p4-cos_shippedweights" >> "$OUT/cos_wk.log" 2>&1
echo "    exit=$? $(date -u +%FT%TZ)"
echo "P4WK_DONE $(date -u +%FT%TZ)"
