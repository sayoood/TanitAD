#!/usr/bin/env bash
# THE LONGITUDINAL ARMS. Every arm is ONE VARIABLE against the SAME named
# baseline `wk15` (ccos, W = 0.0,15.11245,64.29715042415070, --plan-seed 0,
# shipped vocabulary), which is already banked -- so no baseline re-run is
# needed and the comparison is window-for-window.
#
# PRIORITY ORDER (a killed queue still yields value):
#   1 lonvocab   + --a-sustain-mode a0        the VOCABULARY lever (headline)
#   2 lonseam    + --jerk-seam a0             the COST lever, alone
#   3 lonvocab_s1  = lonvocab, --plan-seed 1  the REPLICATE. Mandatory before
#                                             any claim: H-ESTIM-SEED-1 /
#                                             D-REFAV1-CG-SEEDFLOOR measured
#                                             `separated` on 4/10 paired family
#                                             metrics between two arms that
#                                             differ ONLY in --plan-seed.
#   4 loncomb    + all three                  the combined arm
#   5 lonvend    + --target-speed-mode a0ext  arms the dead third weight alone
#
# GATE: distinct `--out` TARGETS, never a process count. M28 (3): one arm is a
# parent and its child (2-4 python.exe entries, ALL carrying the full command
# line), so a gate on "processes <= 1" can never open. The artifact is the arm.
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
        --no-lead-block --cost-metric ccos
        --cost-weights 0.0,15.11245,64.29715042415070)

# how many DISTINCT arms are live? (distinct --out targets among python.exe)
n_arms () {
  powershell.exe -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { if(\$_.CommandLine -match '--out\s+(\S+)'){ \$Matches[1] } } | Sort-Object -Unique).Count" \
    2>/dev/null | tr -d '\r ' | tr -dc '0-9'
}

wait_slot () {                       # block until < 2 distinct arms are live
  for i in $(seq 1 900); do
    n=$(n_arms); n=${n:-9}
    [ "$n" -lt 2 ] && { echo "ZZSLOT-OPEN-n${n}-i${i}ZZ $(date -u +%FT%TZ)"; return 0; }
    [ $((i % 10)) -eq 0 ] && echo "ZZWAIT-i${i}-arms${n}ZZ $(date -u +%FT%TZ)"
    sleep 30
  done
  echo "ZZSLOT-TIMEOUTZZ"; return 1
}

run () {
  local tag="$1"; shift
  wait_slot || return 1
  rm -rf "$OUT/dump_$tag"
  echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
  "$PY" "$M/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
    --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
    --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
  echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
}

run lonshift     --plan-seed 0 --a-sustain-mode a0_shift
run lonseam     --plan-seed 0 --jerk-seam a0
run lonshift_s1  --plan-seed 1 --a-sustain-mode a0_shift
run loncomb2     --plan-seed 0 --a-sustain-mode a0_shift --jerk-seam a0
run lonvocab     --plan-seed 0 --a-sustain-mode a0
echo "ZZQUEUELON2-DONE-$(date -u +%FT%TZ)ZZ"
