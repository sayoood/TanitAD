#!/usr/bin/env bash
# CORRECTED longitudinal queue. Supersedes queueLON2.sh, which queued a seam arm
# that was INERT BY CONSTRUCTION.
#
# ⛔ WHAT WENT WRONG AND IS FIXED HERE. `lonseam` was run as ONE VARIABLE against
# `wk15`, whose triple is `(0.0, 15.11245, 64.29715042415070)` -- i.e. W_JERK = 0.
# The cost is `c + w_jerk * mean(jerk^2)`, so repairing `jerk` while `w_jerk` is
# zero cannot change the objective by a single bit. The arm ran to completion,
# the reached-it guard passed (the record carries `jerk_seam: "a0"`), and it
# banked `+0.0000 [+0.0000, +0.0000]` on all ten family metrics with emitted
# controls bit-identical to `wk15`. That is arithmetic, not a null about the
# seam. `refav1_arm.py` now REFUSES the combination before the rollout.
# ⇒ the informative form is a PAIR that share a non-zero W_JERK and differ ONLY
# in the seam: `seambase` (seam off) vs `seamon` (seam on).
#
# ⛔ `loncomb2` is DROPPED: it carried the same inert seam half, so it would have
# been `lonshift` re-run under a name claiming two levers.
#
# GATE: distinct `--out` TARGETS, never a process count (M28 (3): one arm is a
# parent AND its child, both carrying the full command line).
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

common=(--ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt
        --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json
        --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n 0 --window-stride 16 --no-navshuf
        --no-lead-block --cost-metric ccos)

WK15=0.0,15.11245,64.29715042415070          # the banked baseline triple
JERK=0.02,15.11245,64.29715042415070         # W_JERK restored to the shipped 0.02

n_arms () {
  powershell.exe -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { if(\$_.CommandLine -match '--out\s+(\S+)'){ \$Matches[1] } } | Sort-Object -Unique).Count" \
    2>/dev/null | tr -d '\r ' | tr -dc '0-9'
}

wait_slot () {
  for i in $(seq 1 900); do
    n=$(n_arms); n=${n:-9}
    [ "$n" -lt 2 ] && { echo "ZZSLOT-OPEN-n${n}-i${i}ZZ $(date -u +%FT%TZ)"; return 0; }
    [ $((i % 10)) -eq 0 ] && echo "ZZWAIT-i${i}-arms${n}ZZ $(date -u +%FT%TZ)"
    sleep 30
  done
  echo "ZZSLOT-TIMEOUTZZ"; return 1
}

run () {
  local tag="$1" w="$2"; shift 2
  wait_slot || return 1
  rm -rf "$OUT/dump_$tag"
  echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
  "$PY" "$M/taniteval/tools/refav1_arm.py" "${common[@]}" --cost-weights "$w" "$@" \
    --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
    --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
  echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
}

# 1. THE REPLICATE. Mandatory before D2's result is quotable without a caveat.
run lonshift_s1 "$WK15" --plan-seed 1 --a-sustain-mode a0_shift
# 2-3. THE SEAM PAIR: same W_JERK = 0.02, the SEAM is the only variable.
run seambase    "$JERK" --plan-seed 0
run seamon      "$JERK" --plan-seed 0 --jerk-seam a0
# 4. D1, for ATTRIBUTION: was the maintain-only form enough, or did D2's extra
#    9 windows carry the win?
run lonvocab    "$WK15" --plan-seed 0 --a-sustain-mode a0
# 5. THE REAL COMBINED ARM -- both levers, on a triple where BOTH can act.
run loncomb3    "$JERK" --plan-seed 0 --a-sustain-mode a0_shift --jerk-seam a0
echo "ZZQUEUELON3-DONE-$(date -u +%FT%TZ)ZZ"
