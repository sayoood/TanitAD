#!/bin/bash
# run_amend6.sh -- AMEND_HWMPLAN_6 steps 2-3 (dev-box CPU; no GPU, no box 2, no training). Needs: the pull done (lat6/CHUNKS.json). The MM launches it AFTER review (launch6.ps1 -Job amend6.cmd).
# Stages (logged to $ROOT/logs): 0 spec sha, 1 K-control + fit + FIT-CURVE + tests, 2 section 7 controls, 3 TRAIN stage (beside), 4 SCORE arms, 5 readout -> readout/RESULT_AMEND6.json + .md.
# Last line of $ROOT/read.log:  ZZ AMEND6 READ DONE ZZ | ZZ AMEND6 READ FAIL ZZ <why>
export PATH="/usr/bin:/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree
export PYTHONPATH="F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
ROOT=F:/TanitAD-scratch/actrsp/lonplan/amend6
B=$ROOT/build
export TMP=$ROOT/work/tmp TEMP=$ROOT/work/tmp
mkdir -p "$TMP" "$ROOT/logs" "$ROOT/controls" "$ROOT/readout" "$ROOT/readout_in"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$ROOT/read.log
fail() { echo "ZZ AMEND6 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
say() { echo "$(date -u +%T) $*" >> "$LOG"; }
gate() { $PY -u "$WC" 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h ($1)"; }
cd "$B" || fail "no build folder"
say "start $(date -u +%FT%TZ)"
want=$(cut -c1-64 "$ROOT/AMEND_HWMPLAN_6.sha256"); got=$(sha256sum "$ROOT/AMEND_HWMPLAN_6.md" | cut -c1-64)
[ "$got" = "$want" ] || fail "AMEND_HWMPLAN_6.md sha256 $got is not the frozen $want"
[ -f "$ROOT/lat6/CHUNKS.json" ] || fail "lat6/CHUNKS.json is missing: run the pull first"
[ -n "$(ls -A "$ROOT/score" 2>/dev/null)" ] && fail "score/ is not empty (the wrapper refuses a used folder)"
[ -n "$(ls -A "$ROOT/train" 2>/dev/null)" ] && fail "train/ is not empty (the wrapper refuses a used folder)"
gate tests
(cd "$B" && $PY -m pytest -q -p no:cacheprovider --basetemp="$ROOT/work/tmp/bt_test_amend6" test_amend6.py) > "$ROOT/logs/tests_test_amend6.log" 2>&1 || fail "tests (logs/tests_test_amend6.log)"
say "tests: $(tail -n 1 "$ROOT/logs/tests_test_amend6.log")"
# ---- 1: K-control (141 clips reproduce AMEND_5's RA), the fit on 141 + the largest complete checkpoint, FIT-CURVE
for k in s0 s1; do
  gate "fit $k"
  $PY -u fit_ra6.py kcontrol --chassis $k > "$ROOT/logs/kcontrol_$k.log" 2>&1 || fail "K-control $k is RED (logs/kcontrol_$k.log)"
  $PY -u fit_ra6.py fit --chassis $k > "$ROOT/logs/fit_$k.log" 2>&1 || fail "fit $k (logs/fit_$k.log)"
  $PY -u fit_ra6.py curve --chassis $k > "$ROOT/logs/curve_$k.log" 2>&1 || fail "FIT-CURVE $k (logs/curve_$k.log)"
done
say "K-control, fit and FIT-CURVE done: $(tail -n 1 "$ROOT/logs/fit_s0.log") | $(tail -n 1 "$ROOT/logs/fit_s1.log")"
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
ctl() { $PY -u amend6_controls.py run --chassis "$1"; }
trn() { $PY -u amend6_run.py plan --stage train --chassis "$1"; }
scr() { $PY -u amend6_run.py plan --stage score --chassis "$1"; }
# ---- 2: controls (every control and its regression arm)
pair controls ctl
$PY -u amend6_controls.py merge >> "$LOG" 2>&1 || fail "a control or its regression arm is RED (controls/CONTROLS.json): INVALID, nothing planned on SCORE"
say "controls GREEN"
# ---- 3: TRAIN stage (beside) + C-SHUF sign + C-PLANT (ii)
pair train trn
for k in s0 s1; do $PY -u amend6_read.py train --chassis $k >> "$LOG" 2>&1 || fail "TRAIN-stage check (C-SHUF sign or C-PLANT ii) failed at $k (readout/TRAIN_STAGE_$k.json)"; done
say "TRAIN stage done"
# ---- 4: SCORE arms
pair score scr
# ---- 5: readout
gate readout
$PY -u amend6_read.py score >> "$LOG" 2>&1 || fail "readout (see $LOG)"
echo "ZZ AMEND6 READ DONE ZZ" | tee -a "$LOG"
