#!/bin/bash
# run_amend5.sh -- AMEND_HWMPLAN_5 (RA-L1) read, dev-box CPU only (no GPU, no box 2, no training). The MM launches it AFTER review and AFTER BASE_SOLVER_AMEND5.md exists (WMI: launch_amend5.ps1 -> amend5.cmd).
# Stages (logged to $ROOT/logs): 0 spec sha + tests, 1 section 8 controls (C-READBACK of every base file used, C-OFF5, mutations on TRAIN windows), 2 TRAIN stage (beside, in-sample; C-SHUF sign, C-PLANT ii),
#   3 SCORE RA arms (REFUSED by ra_l1 / amend5_run unless BASE_SOLVER_AMEND5.md names the branch), 4 readout (contrasts, labels, four families) -> readout/RESULT_AMEND5.json + .md.
# No RA plan exists on a SCORE window before controls are GREEN and the TRAIN stage checks hold at both chassis.
# Chassis run side by side when >= 11 GB of commit is free, else one after the other, each gated at >= 5.5 GB.
# Last line of $ROOT/read.log:  ZZ AMEND5 READ DONE ZZ  |  ZZ AMEND5 READ FAIL ZZ <why>
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
export WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree
export PYTHONPATH="F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
ROOT=${AMEND5_ROOT:-F:/TanitAD-scratch/actrsp/lonplan/amend5}
export AMEND5_ROOT=$ROOT
B=F:/TanitAD-scratch/actrsp/lonplan/amend5/build
export TMP=$B/tmp TEMP=$B/tmp
mkdir -p "$TMP" "$ROOT/logs" "$ROOT/controls" "$ROOT/readout"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$ROOT/read.log
TH=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory
fail() { echo "ZZ AMEND5 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
say() { echo "$(date -u +%T) $*" >> "$LOG"; }
gate() { $PY -u "$WC" 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h ($1)"; }
cd "$B" || fail "no build folder"
say "start $(date -u +%FT%TZ)"
# ---- 0: the frozen spec, the base solver file, the folders, the tests
want=$(cut -c1-64 "$ROOT/AMEND_HWMPLAN_5.sha256")
got=$(sha256sum "$ROOT/AMEND_HWMPLAN_5.md" | cut -c1-64)
[ "$got" = "$want" ] || fail "AMEND_HWMPLAN_5.md sha256 $got is not the frozen $want"
[ -f "$ROOT/BASE_SOLVER_AMEND5.md" ] || fail "BASE_SOLVER_AMEND5.md is missing: the MM writes it from AMEND_HWMPLAN_4's result (lines BASE_SOLVER: A0|P|P23 and BRANCH: B1|B2|B3) before any SCORE RA plan"
$PY -c "import sys; sys.path.insert(0,'$B'); import amend5_stage as S; print(S.read_base_solver())" >> "$LOG" 2>&1 || fail "BASE_SOLVER_AMEND5.md is unreadable or inconsistent"
[ -n "$(ls -A "$ROOT/score" 2>/dev/null)" ] && fail "score/ is not empty (the wrapper refuses a used folder)"
gate tests
for t in "test_amend5.py:$B" "test_lonplan.py:F:/TanitAD-scratch/actrsp/lonplan/build" "test_hwm_plan.py:$TH/hwm_plan" "test_plantest2_controls.py:$TH/plantest2" "test_amend4.py:F:/TanitAD-scratch/actrsp/lonplan/amend4/build"; do
  f=${t%%:*}; dir=${t#*:}
  [ -f "$dir/$f" ] || { say "tests $f: not found at $dir (skipped)"; continue; }
  # MM 2026-11-17: test_plantest2_controls asserts tanitad comes from PLANTEST's pinned export (plantest_real.py:13), not ladder_tree.
  # It runs in that tree (F:/TanitAD-scratch/plantest_tree = git archive 26fb0bc1 stack/tanitad taniteval/taniteval, re-exported by the MM).
  TENV=()
  if [ "$f" = test_plantest2_controls.py ]; then
    [ "$(git -C C:/Users/Admin/tanitad-push rev-parse 26fb0bc1:stack/tanitad/__init__.py)" = "$(git hash-object F:/TanitAD-scratch/plantest_tree/stack/tanitad/__init__.py)" ] || fail "plantest_tree is missing or not the 26fb0bc1 export"
    TENV=(PYTHONPATH="F:\TanitAD-scratch\plantest_tree\stack;F:\TanitAD-scratch\plantest_tree\taniteval" WMACT_CODE_ROOT=F:/TanitAD-scratch/plantest_tree)
  fi
  (cd "$dir" && env "${TENV[@]}" $PY -m pytest -q -p no:cacheprovider --basetemp="$B/tmp/bt_${f%.py}" "$f") > "$ROOT/logs/tests_${f%.py}.log" 2>&1 || fail "tests $f (logs/tests_${f%.py}.log)"
  say "tests $f: $(tail -n 1 "$ROOT/logs/tests_${f%.py}.log")"
done
pair() {
  name=$1; shift
  if $PY -u "$WC" 11 1 >> "$LOG" 2>&1; then
    for k in s0 s1; do ( "$@" $k > "$ROOT/logs/${name}_$k.log" 2>&1; echo $? > "$ROOT/logs/${name}_$k.rc" ) & done
    wait
  else
    for k in s0 s1; do gate "$name $k"; "$@" $k > "$ROOT/logs/${name}_$k.log" 2>&1; echo $? > "$ROOT/logs/${name}_$k.rc"; done
  fi
  for k in s0 s1; do [ "$(cat "$ROOT/logs/${name}_$k.rc")" = "0" ] || fail "$name $k rc $(cat "$ROOT/logs/${name}_$k.rc") (logs/${name}_$k.log)"; done
  say "$name done"
}
# bases whose SCORE files exist (A0 always; P / P23 from AMEND_4 with l2_prim_raw)
BASES="A0"
for x in P P23; do [ -f "${AMEND4_ROOT:-F:/TanitAD-scratch/actrsp/lonplan/amend4}/score/${x}_s0/plans_hwm_oracle_s0.npz" ] && BASES="$BASES $x"; done
say "bases for the controls: $BASES"
ctl() { $PY -u amend5_controls.py run --chassis "$1" --bases $BASES $( [ "$1" = s0 ] && echo --cstrat ); return 0; }   # RED is read by the merge; a crash leaves no controls_<k>.json and the merge fails
trn() { $PY -u amend5_run.py plan --stage train --chassis "$1" --bases $BASES; }
scr() { $PY -u amend5_run.py plan --stage score --chassis "$1"; }
# ---- 1: controls
pair controls ctl
$PY -u amend5_controls.py merge >> "$LOG" 2>&1 || fail "a section 8 control or its regression arm is RED (controls/CONTROLS.json): INVALID, nothing planned on SCORE"
say "controls GREEN"
# ---- 2: TRAIN stage (beside) + C-SHUF sign + C-PLANT (ii)
pair train trn
for k in s0 s1; do $PY -u amend5_read.py train --chassis $k >> "$LOG" 2>&1 || fail "TRAIN-stage check (C-SHUF sign or C-PLANT ii) failed at $k (readout/TRAIN_STAGE_$k.json)"; done
say "TRAIN stage done; C-SHUF sign and C-PLANT (ii) hold at both chassis"
# ---- 3: SCORE RA arms
pair score scr
# ---- 4: readout
gate readout
$PY -u amend5_read.py score >> "$LOG" 2>&1 || fail "readout (see $LOG)"
echo "ZZ AMEND5 READ DONE ZZ" | tee -a "$LOG"
