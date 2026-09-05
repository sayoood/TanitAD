#!/usr/bin/env bash
# P4-C — THE FIX, ON THE GPU: does a finer SUSTAINED goal curvature make refav1
# track a real road curve?
#
# Runs AFTER gm_run_p4.sh's three arms, on the SAME 8-episode turn-dense panel
# and the SAME window grid, so it is paired with them window-for-window.
#
#   C  ccos + argmax + --goal-kappa-turn 0.02   (R 50 m)
#
# ⭐ WHY 0.02 AND NOT SOMETHING ELSE. MEASURED, zero GPU, and committed before
# this ran (`ESCALATION.md`, `raw/vocab_design.json`, 4520 windows / 141
# episodes): under an ORACLE token chooser kappa_turn=0.02 makes 100 % of real
# turns EXPRESSIBLE against 38.7 % at the shipped 0.08, and cuts the MEDIAN
# curvature error on a turn 2.5x (0.01551 -> 0.00609) against a LANE_KEEP-only
# floor of 0.01942. It is the best SINGLE magnitude on the median turn; L=3
# levels would be better still but is a vocabulary change, not a constant.
#
# ⛔ ccos IS MANDATORY HERE, not a preference: under the shipped `cos` a
# correctly decoded TURN is refused on 38/38 real windows (D-REFAV1-DRIVE-GATE2),
# so a finer goal curvature under `cos` would be measured as zero for a reason
# that has nothing to do with the vocabulary.
#
# ⚠️ WHAT A NULL WOULD MEAN, committed in advance. A finer goal moves the FIELD
# the search is scored against; whether iCEM then tracks it is a separate
# question the zero-GPU work explicitly does NOT answer (the lookup shortcut does
# not hold on turn windows). So:
#   * executed |kappa| moves toward 0.02 on GT-turn windows => the vocabulary was
#     the binding constraint at plan level, as the zero-GPU work predicts;
#   * executed |kappa| unchanged (still 0.08 or still 0) => the binding
#     constraint at plan level is the SEARCH or the cost, not the vocabulary,
#     and the escalation must be re-scoped. That is a real outcome, not a
#     failure of the experiment.
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_margin/p4out"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"

# wait for the P4 driver's own arms to finish (poll for ABSENCE, max 3 h)
for i in $(seq 1 1080); do
  live=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -like '*refav1_arm.py*' }).Count" 2>/dev/null | tr -d '\r ')
  # NO '|| echo 0' HERE. grep -c prints 0 AND exits 1 when there are no
  # matches, so $(grep -c ... || echo 0) captures TWO lines of 0, which
  # then passes a string test against 0 and starts this arm CONCURRENTLY
  # with the P4 driver's own, doubling GPU load on an 8 GB card. Caught
  # before it fired. The previous version was KILLED and replaced under a
  # new filename, never edited in place: bash reads a running script
  # lazily by byte offset.
  done_marker=$(grep -c "P4_ALL_ARMS_DONE" "$OUT/driver.log" 2>/dev/null)
  done_marker=$(printf '%s' "${done_marker:-0}" | tr -dc '0-9')
  live=$(printf '%s' "${live:-1}" | tr -dc '0-9')
  echo "[waitC $i] refav1_arm procs=${live:-?} p4_done=${done_marker:-?} $(date -u +%FT%TZ)"
  if [ "${done_marker:-0}" -ge 1 ] 2>/dev/null && [ "${live:-1}" -eq 0 ] 2>/dev/null; then break; fi
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
  --no-lead-block --cost-metric ccos \
  --cost-weights 0.0,0.0,64.29715042415070 \
  --goal-kappa-turn 0.02 --plan-seed 0 \
  --dump-dir "$OUT/dump_ccos_k002" --out "$OUT/rec_ccos_k002.json" \
  --arm "refav1-21109-p4-ccos_k002" >> "$OUT/ccos_k002.log" 2>&1
echo "    exit=$? $(date -u +%FT%TZ)"
echo "P4C_DONE $(date -u +%FT%TZ)"
