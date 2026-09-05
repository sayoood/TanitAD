#!/usr/bin/env bash
# P1 — THE DE-CONFOUNDED ORACLE. One process, three planning arms, one window
# grid, one plan seed, so the goal is the only thing that moves.
#
#   cl             the shipped arm            (goal = the head's imagination)
#   cl_oraclegoal  the CONFOUNDED banked arm  (goal = true future, NO seed)
#   cl_oracleseed  the DE-CONFOUNDED arm      (goal = true future, seed KEPT)
#
# Panel: the goal-margin stream's TURN-ENRICHED p4 set — 8 episodes, 40 windows
# at stride 16, 17 of them real turns at the MEASURED crossover |gt_k| > 4e-2.
# A turning-window question needs turning windows; the `abt` grid has 28.
#
# ⛔ Waits for the sibling p4 `cos` job to exit first: the 4060 has ~3.1 GB free
# of 8.2 and three python jobs already on it. Adding a fourth risks an OOM that
# would read as a failed experiment.
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_drive/oracle"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"

# --- wait for the sibling job by EXPLICIT PID (never pgrep -f: it self-matches)
for i in $(seq 1 120); do
  n=$(powershell.exe -NoProfile -Command \
      "@(Get-Process -Id 30120,36688 -ErrorAction SilentlyContinue).Count" \
      2>/dev/null | tr -d '\r ')
  [ "${n:-0}" = "0" ] && { echo "[wait] sibling gone after ${i} polls"; break; }
  echo "[wait] $i sibling alive n=$n $(date -u +%FT%TZ)"
  sleep 30
done

LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CACHE=C:/Users/Admin/refav1_margin/p4/fp8
EPS=C:/Users/Admin/refav1_margin/p4/eps

common=(--ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt
        --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json
        --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n 0 --window-stride 16 --no-navshuf
        --no-lead-block --cost-metric ccos
        --cost-weights 0.0,0.0,64.29715042415070
        --with-oracle-goal-arm --with-oracle-seed-arm)

run () { local name="$1"; shift
  echo "=== ARM $name  $(date -u +%FT%TZ) ==="
  "$PY" "$M/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
      --dump-dir "$OUT/dump_$name" --out "$OUT/rec_$name.json" \
      --arm "refav1-21109-p4-$name" >> "$OUT/$name.log" 2>&1
  echo "    exit=$? $(date -u +%FT%TZ)"; }

run oracle_s0 --plan-seed 0
echo "ORACLE_S0_DONE $(date -u +%FT%TZ)"
run oracle_s1 --plan-seed 1
echo "ORACLE_DONE $(date -u +%FT%TZ)"
