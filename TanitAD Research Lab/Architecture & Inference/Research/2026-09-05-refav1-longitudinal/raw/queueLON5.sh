#!/usr/bin/env bash
# CORRECTED longitudinal queue, v4. Supersedes queueLON3.sh (a gate that counted
# a non-arm process) and queueLON2.sh (a seam arm inert by construction).
#
# ⛔ FIX 1 (v3): `lonseam` was run as one variable against `wk15`, whose triple is
# `(0.0, ...)` -- W_JERK = 0. The cost adds `w_jerk * mean(jerk^2)`, so repairing
# `jerk` while `w_jerk` is zero cannot change the objective by a single bit. The
# arm banked +0.0000 on all ten metrics: arithmetic, not a null. `refav1_arm.py`
# now REFUSES that combination before the rollout. ⇒ the seam is measured as a
# PAIR sharing a live weight: `seambase` (seam off) vs `seamon` (seam on).
#
# ⛔ FIX 2 (v4): the v3 GATE counted a sibling stream's ANALYSIS script, which
# also uses `--out` (`--out ../raw/l3_splitp30k.json`), as an arm -- so it read
# "4 arms" while 2 were live and `< 2` could never be satisfied. Counting the
# artifact was right; counting `--out` alone was not specific enough.
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

# COUNT THE ARM ARTIFACT, AND ONLY THE ARM'S. Three traps avoided, all measured:
#  1. M28 (3): one arm is a parent AND its child, both carrying the full command
#     line, so a PROCESS count never falls below 2 while an arm runs.
#  2. `--out` is not unique to the arm tool -- require the TOOL NAME too.
#  3. The tool name is assembled from two pieces, so this script's own command
#     line never contains the literal pattern it searches for (the pgrep
#     self-match trap, which cost a killed shell earlier this turn).
n_arms () {
  powershell.exe -NoProfile -Command "\$t = '*refav1' + '_arm.py*'; @(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like \$t } | ForEach-Object { if(\$_.CommandLine -match '--out\s+(\S+)'){ \$Matches[1] } } | Sort-Object -Unique).Count" 2>/dev/null | tr -d '\r ' | tr -dc '0-9'
}

wait_slot () {
  for i in $(seq 1 1200); do
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

# ==== v5 RE-RANK, on evidence that arrived after v4 was queued ============ #
# The Thor stream landed a full panel (their commit 2f494aa) that changes what
# is worth running here:
#   * they ran `T_lonvocab` and answered D1-vs-D2 (indistinguishable at arm
#     level) => `lonvocab` is DROPPED from this queue as redundant;
#   * they measured `a_shift`'s OWN inference-seed floor on Thor and report it
#     replicating across two GPUs to 0.002 m/s => `lonshift_s1` drops to LAST,
#     kept only because the seed floor is now known to be RIG-DEPENDENT;
#   * they confirmed the seam is inert at W_JERK = 0 on a second rig and called
#     the successor "BLOCKED BY CONSTRUCTION at these weights" -- which is
#     exactly what the pair below unblocks, and NOBODY has run it.
#
# ⭐ AND THE PAIR IS PRICED BEFORE IT RUNS (raw/lon_jerk_scale.txt, zero GPU):
# at W_JERK = 0.02 the seam term's median contribution is HALF the goal+kappa
# terms combined on `wk15` (ratio 0.494) and 4.05x LARGER on `lonshift`. So it
# is first-order, not "technically live, practically inert" -- the trap one
# notch weaker that would have cost two more GPU hours to discover.
# ⚠ COMMITTED OUTCOME, before the numbers: at ratio 4.05 the seam could
# DOMINATE D2's objective in `loncomb3` and undo the vocabulary win. If
# `loncomb3` is worse than `lonshift`, that is the finding, reported as such --
# not a reason to retune the weight.

# 1-2. THE SEAM PAIR: same live W_JERK = 0.02, the SEAM the only variable.
run seambase    "$JERK" --plan-seed 0
run seamon      "$JERK" --plan-seed 0 --jerk-seam a0
# 3. BOTH LEVERS, on a triple where both can act.
run loncomb3    "$JERK" --plan-seed 0 --a-sustain-mode a0_shift --jerk-seam a0
# 4. This rig's OWN replicate of D2 -- last, because Thor already replicated the
#    effect across GPUs, but kept because the seed floor is rig-dependent.
run lonshift_s1 "$WK15" --plan-seed 1 --a-sustain-mode a0_shift
echo "ZZQUEUELON5-DONE-$(date -u +%FT%TZ)ZZ"
