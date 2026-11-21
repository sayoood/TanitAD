#!/bin/bash
# reread_dtfix.sh -- MM 4a3d05, 2026-11-12: the E2 judge re-run with ONE change: r9_e2_families.py v2 (dt from the dump's grid.dt_s, 0.5 s) instead of
# v1 (hard-coded 0.1 s).  Same dumps (pulled from Thor, sha256-verified, 866 files), same distkeep / inputs / verdict code (tip 0bbdc066), same EVAL and
# S6c files as the banked read.  Commands = r9e2_roll.sh's CPU stage, step for step.  Writes under reread_dtfix/read/.
export PATH="/usr/bin:/bin:$PATH"; set -u
B=C:/Users/Admin/r9e2_read_K2M1N20; SRC=$B/reread_dtfix/src; RD=$B/reread_dtfix/read; TR=F:/refcv9_e3/famfix_mm/tree
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; SPLIT=$SRC/DEV_HOLD_SPLIT.json; LB=$SRC/b1_eval_lead_block_k4_dt0p5_psync.npz
mkdir -p $RD/s6c $RD/logs
cp $B/pull/read/EVAL_K2.json $B/pull/read/EVAL_K0.json $RD/ && cp $B/pull/read/s6c/S6C_K*_s*.json $RD/s6c/ || exit 3
[ "$(git hash-object --no-filters $TR/stack/scripts/r9_e2_families.py)" = "$(git hash-object --no-filters C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/famfix/stack/scripts/r9_e2_families.py)" ] || { echo "tree reader is not the fixed one"; exit 3; }
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
export PYTHONPATH="$(cygpath -w "$TR/stack");$(cygpath -w "$TR/taniteval");$(cygpath -w "$TR/stack/scripts")"
cd $RD || exit 3
for S in 0 1; do
  $PY -X utf8 $TR/stack/scripts/r9_e2_families.py --mode families --a-dump $SRC/E2_K2_pf/dump_s$S --b-dump $SRC/E2_K0_pf/dump_s$S --dev-split $SPLIT \
      --out $RD/families_s$S.json > $RD/logs/families_s$S.log 2>&1 < /dev/null || { echo "FAMILIES $S"; tail -n 5 $RD/logs/families_s$S.log; exit 4; }
  $PY -X utf8 $TR/stack/scripts/r9_e2_distkeep.py --a-dump $SRC/E2_K2_pf/dump_s$S --b-dump $SRC/E2_K0_pf/dump_s$S --dev-split $SPLIT --lead-block $LB \
      --out $RD/distkeep_s$S.json --merge-into $RD/families_s$S.json > $RD/logs/distkeep_s$S.log 2>&1 < /dev/null || { echo "DISTKEEP $S"; tail -n 5 $RD/logs/distkeep_s$S.log; exit 4; }
  $PY -X utf8 $TR/stack/scripts/r9_e2_families.py --mode camblind --a-dump $SRC/E2_K2_pf/dump_s$S --b-dump $SRC/E2_K2_camblind/dump_s$S --dev-split $SPLIT \
      --out $RD/camblind_s$S.json > $RD/logs/camblind_s$S.log 2>&1 < /dev/null || { echo "CAMBLIND $S"; tail -n 5 $RD/logs/camblind_s$S.log; exit 4; }
  echo "seed $S scored"
done
$PY -X utf8 $TR/stack/ops/r9e2_read/r9e2_inputs.py --rd $RD --out $RD/INPUTS.json --k2-label K2 --k0-label K0 --s6c-dir $RD/s6c > $RD/logs/inputs.log 2>&1 < /dev/null || { echo INPUTS; tail -n 5 $RD/logs/inputs.log; exit 4; }
$PY -X utf8 $TR/stack/scripts/r9_e2_verdict.py --inputs $RD/INPUTS.json --out-dir $RD > $RD/logs/verdict.log 2>&1 < /dev/null; echo "VERDICT rc $?"
tail -n 3 $RD/logs/verdict.log
