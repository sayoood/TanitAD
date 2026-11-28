#!/bin/bash
# run_amend4.sh -- AMEND_HWMPLAN_4 (L2-IMG) read, dev-box CPU only (no GPU, no box 2, no training). The MM launches it AFTER review (WMI: launch_amend4.ps1 -> amend4.cmd).
# Stages (each logged to $ROOT/logs): 0 spec sha + tests, 1 section 8 controls (C-OFF on A0 re-plans batch 0 of the banked SCORE plans; mutations on 16 TRAIN windows), 2 TRAIN stage (P, P23, RS on 564 TRAIN windows; C-RS sign),
#   3 SCORE (P, P23: seeds 0 1; RS: seed 0; both chassis), 4 C-READBACK of every plan file, 5 readout (contrasts, 2-0 rows, DIR-ACC, mechanism rows, labels, four-family tables).
# The launcher REFUSES to plan any SCORE window under prim / prim23 unless CONTROLS.json is green and the TRAIN stage read RS - A0 > 0 at both chassis.
# Chassis run side by side when >= 11 GB of commit is free, else one after the other, each gated at >= 5.5 GB (LON-PLAN's rule; the 11-16 18:02Z stack OOM).
# Last line of $ROOT/read.log:  ZZ AMEND4 READ DONE ZZ  |  ZZ AMEND4 READ FAIL ZZ <why>
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
export WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree
export PYTHONPATH="F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
ROOT=${AMEND4_ROOT:-F:/TanitAD-scratch/actrsp/lonplan/amend4}
export AMEND4_ROOT=$ROOT
B=F:/TanitAD-scratch/actrsp/lonplan/amend4/build
export TMP=$B/tmp TEMP=$B/tmp
mkdir -p "$TMP" "$ROOT/logs" "$ROOT/controls" "$ROOT/readout"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$ROOT/read.log
fail() { echo "ZZ AMEND4 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
say() { echo "$(date -u +%T) $*" >> "$LOG"; }
gate() { $PY -u "$WC" 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h ($1)"; }
cd "$B" || fail "no build folder"
say "start $(date -u +%FT%TZ)"
# ---- 0: the frozen spec, the folders, the tests
want=$(cut -c1-64 "$ROOT/AMEND_HWMPLAN_4.sha256")
got=$(sha256sum "$ROOT/AMEND_HWMPLAN_4.md" | cut -c1-64)
[ "$got" = "$want" ] || fail "AMEND_HWMPLAN_4.md sha256 $got is not the frozen $want"
for d in train score; do
  for x in P P23 RS; do for k in s0 s1; do
    [ -n "$(ls -A "$ROOT/$d/${x}_$k" 2>/dev/null)" ] && fail "$d/${x}_$k is not empty (the wrapper refuses a used folder)"
  done; done
done
gate tests
for t in "test_amend4.py:$B" "test_lonplan.py:F:/TanitAD-scratch/actrsp/lonplan/build" "test_hwm_plan.py:E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan"; do
  f=${t%%:*}; dir=${t#*:}
  (cd "$dir" && $PY -m pytest -q -p no:cacheprovider --basetemp="$B/tmp/bt_${f%.py}" "$f") > "$ROOT/logs/tests_${f%.py}.log" 2>&1 || fail "tests $f (logs/tests_${f%.py}.log)"
  say "tests $f: $(tail -n 1 "$ROOT/logs/tests_${f%.py}.log")"
done
# ---- helper: run "$1 <k>" for s0 and s1, side by side if commit allows, else one after the other
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
ctl() { $PY -u amend4_controls.py run --chassis "$1"; return 0; }   # RED is read by the merge below; a crash leaves no controls_<k>.json and the merge fails
trn() { for x in P P23 RS; do $PY -u amend4_stage.py plan --stage train --arm $x --chassis "$1" || return 1; done; }
scr() { for x in P P23 RS; do $PY -u amend4_stage.py plan --stage score --arm $x --chassis "$1" || return 1; done; }
# ---- 1: controls (no SCORE window is planned under prim / prim23 before they are green)
pair controls ctl
$PY -u amend4_controls.py merge >> "$LOG" 2>&1 || fail "a section 8 control or its regression arm is RED (controls/CONTROLS.json): INVALID, nothing planned on SCORE"
say "controls GREEN"
# ---- 2: TRAIN stage (beside) and the C-RS sign check
pair train trn
for k in s0 s1; do $PY -u amend4_read.py train --chassis $k >> "$LOG" 2>&1 || fail "C-RS TRAIN sign: RS - A0 is not > 0 at $k (readout/TRAIN_STAGE_$k.json)"; done
say "TRAIN stage done, C-RS sign holds at both chassis"
# ---- 3: SCORE
pair score scr
for k in s0 s1; do for x in P P23 RS; do cmp -s "$ROOT/score/${x}_$k/windows.npz" "F:/TanitAD-scratch/actrsp/lonplan/score/A0_$k/windows.npz" || fail "score/${x}_$k windows differ from A0_$k"; done; done
# ---- 4: C-READBACK on every plan file (10), batch 0 from SOLVER.json at 2 threads
rbk() { for x in P P23 RS; do $PY -u amend4_run.py readback --out "$ROOT/score/${x}_$1" || return 1; done; }
pair readback rbk
# ---- 5: readout
gate readout
$PY -u amend4_read.py score >> "$LOG" 2>&1 || fail "readout (see $LOG)"
echo "ZZ AMEND4 READ DONE ZZ" | tee -a "$LOG"
