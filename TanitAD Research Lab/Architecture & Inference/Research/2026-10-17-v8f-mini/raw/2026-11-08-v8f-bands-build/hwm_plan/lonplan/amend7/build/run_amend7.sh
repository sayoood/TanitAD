#!/bin/bash
# run_amend7.sh -- AMEND_HWMPLAN_7 (dev-box CPU; no GPU, no box 2, no world-model training). Usage: run_amend7.sh fit | score
#   fit    steps 0-4 of the chain: spec sha, tests, pair tables, cross-fits (RA KCTL CONST MLPA MLPB PLANT SHUFT ZERO), final readouts A and B, GAIN7, the FIT controls, gate G7-FIT. Ends after the gate.
#          Last line of $ROOT/read.log:  ZZ AMEND7 FIT DONE ZZ <outcome> | ZZ AMEND7 FIT INVALID ZZ <control> | ZZ AMEND7 FIT FAIL ZZ <why> | ZZ AMEND7 FAIL ZZ spec
#   score  step 5 (only after R7-FIT-PASS; the MM decides): controls, TRAIN stage, SCORE arms, readout.  Last line: ZZ AMEND7 READ DONE ZZ <outcome> | ZZ AMEND7 READ FAIL ZZ <why>
# The two chassis run ONE AFTER THE OTHER (AMEND7_PARALLEL=1 runs them side by side only with >= 11 GB commit headroom). Repair notes never go into read.log (watchers match its text).
MODE=${1:-fit}
export PATH="/usr/bin:/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export WMACT_CODE_ROOT=F:/TanitAD-scratch/ladder_tree
export PYTHONPATH="F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
ROOT=F:/TanitAD-scratch/actrsp/lonplan/amend7
B=$ROOT/build
export TMP=$ROOT/work/tmp TEMP=$ROOT/work/tmp
mkdir -p "$TMP" "$ROOT/logs" "$ROOT/controls" "$ROOT/readout" "$ROOT/readout_in" "$ROOT/fit" "$ROOT/pairs"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$ROOT/read.log
if [ "$MODE" = "score" ]; then FAILTXT="READ FAIL"; else FAILTXT="FIT FAIL"; fi
fail() { echo "ZZ AMEND7 $FAILTXT ZZ $1" | tee -a "$LOG"; exit 1; }
say() { echo "$(date -u +%T) $*" >> "$LOG"; }
gate() { $PY -u "$WC" 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h ($1)"; }
cd "$B" || fail "no build folder"
say "start $MODE $(date -u +%FT%TZ)"
want=$(cut -c1-64 "$ROOT/AMEND_HWMPLAN_7.sha256"); got=$(sha256sum "$ROOT/AMEND_HWMPLAN_7.md" | cut -c1-64)
[ "$got" = "$want" ] || { echo "ZZ AMEND7 FAIL ZZ spec" | tee -a "$LOG"; exit 1; }
each_chassis() {                     # each_chassis NAME FN : FN s0, then FN s1 (serial; parallel only with AMEND7_PARALLEL=1 and >= 11 GB)
  name=$1; fn=$2
  if [ "$AMEND7_PARALLEL" = "1" ] && $PY -u "$WC" 11 1 >> "$LOG" 2>&1; then
    for k in s0 s1; do ( $fn $k > "$ROOT/logs/${name}_$k.log" 2>&1; echo $? > "$ROOT/logs/${name}_$k.rc" ) & done
    wait
  else
    for k in s0 s1; do
      gate "$name $k"
      $fn $k > "$ROOT/logs/${name}_$k.log" 2>&1; echo $? > "$ROOT/logs/${name}_$k.rc"
      [ "$(cat "$ROOT/logs/${name}_$k.rc")" = "0" ] || break
    done
  fi
  for k in s0 s1; do [ "$(cat "$ROOT/logs/${name}_$k.rc" 2>/dev/null)" = "0" ] || fail "$name $k rc $(cat "$ROOT/logs/${name}_$k.rc" 2>/dev/null) (logs/${name}_$k.log)"; done
  say "$name done"
}
fitchas() {                          # one chassis: pair table, cross-fits, final readouts, GAIN7 (a step whose output exists is skipped: delete fit/ to force)
  k=$1
  $PY -u pairs7.py --chassis $k || return 1
  for arm in RA KCTL CONST MLPA MLPB PLANT SHUFT ZERO; do
    if [ -f "$ROOT/fit/${arm}_$k.json" ]; then echo "skip $arm $k (fit/${arm}_$k.json exists)"; continue; fi
    gate "crossfit $arm $k"
    $PY -u fit7.py crossfit --chassis $k --arm $arm || return 1
  done
  for ss in A B; do
    suf=""; [ "$ss" = "B" ] && suf="_B"
    if grep -q "\"sha256$suf\"" "$ROOT/RA7_FIT_$k.json" 2>/dev/null; then echo "skip final $ss $k"; continue; fi
    gate "final $ss $k"
    $PY -u fit7.py final --chassis $k --seed-set $ss || return 1
  done
  $PY -u fit7.py gain --chassis $k || return 1
}
fitctl() { $PY -u amend7_controls.py fit --chassis "$1"; }
runctl() { $PY -u amend7_controls.py run --chassis "$1"; }
trn() { $PY -u amend7_run.py plan --stage train --chassis "$1"; }
scr() { $PY -u amend7_run.py plan --stage score --chassis "$1"; }
bytes() { du -sb "$1" 2>/dev/null | cut -f1; }

if [ "$MODE" = "fit" ]; then
  gate tests
  (cd "$B" && $PY -m pytest -q -p no:cacheprovider --basetemp="$ROOT/work/tmp/bt_test_amend7" test_amend7.py) > "$ROOT/logs/tests_test_amend7.log" 2>&1 || fail "tests (logs/tests_test_amend7.log)"
  say "tests: $(tail -n 1 "$ROOT/logs/tests_test_amend7.log")"
  each_chassis fit fitchas
  each_chassis fitctl fitctl
  if ! $PY -u amend7_controls.py merge --which fit >> "$ROOT/logs/merge_fit.log" 2>&1; then
    red=$($PY -c "import json;print(','.join(json.load(open('$ROOT/controls/CONTROLS_FIT.json'))['red']))" | tr -d '\r')
    echo "ZZ AMEND7 FIT INVALID ZZ $red" | tee -a "$LOG"; exit 1
  fi
  say "FIT controls GREEN"
  gate gate
  $PY -u fit7.py gate > "$ROOT/logs/gate.log" 2>&1 || fail "gate (logs/gate.log)"
  out=$($PY -c "import json;print(json.load(open('$ROOT/readout/G7_FIT.json'))['outcome'])" | tr -d '\r')
  # the re-creatable pair tables and the work folder are deleted at the end of the finished chain; each deletion is logged with its size
  for d in "$ROOT/pairs" "$ROOT/work"; do say "deleted $d ($(bytes "$d") bytes)"; done
  rm -rf "$ROOT/pairs" "$ROOT/work"
  echo "ZZ AMEND7 FIT DONE ZZ $out" | tee -a "$LOG"
  exit 0
fi

# ---- score mode: only after R7-FIT-PASS
lbl=$($PY -c "import json;print(json.load(open('$ROOT/readout/G7_FIT.json'))['outcome_label'])" 2>/dev/null | tr -d '\r')
[ "$lbl" = "R7-FIT-PASS" ] || fail "the FIT outcome is '$lbl', not R7-FIT-PASS: the SCORE read does not run"
[ -n "$(ls -A "$ROOT/score" 2>/dev/null)" ] && fail "score/ is not empty (the wrapper refuses a used folder)"
[ -n "$(ls -A "$ROOT/train" 2>/dev/null)" ] && fail "train/ is not empty (the wrapper refuses a used folder)"
mkdir -p "$ROOT/work/tmp"
each_chassis ctl runctl
$PY -u amend7_controls.py merge --which run >> "$LOG" 2>&1 || fail "a control or its regression arm is RED (controls/CONTROLS.json): INVALID, nothing planned on SCORE"
say "controls GREEN"
each_chassis train trn
for k in s0 s1; do $PY -u amend7_read.py train --chassis $k >> "$LOG" 2>&1 || fail "TRAIN-stage check (C-SHUF7 sign or C-PLANT7 ii) failed at $k (readout/TRAIN_STAGE_$k.json)"; done
say "TRAIN stage done"
each_chassis score scr
gate readout
$PY -u amend7_read.py score > "$ROOT/logs/read_score.log" 2>&1 || fail "readout (logs/read_score.log)"
out=$($PY -c "import json;print(json.load(open('$ROOT/readout/RESULT_AMEND7.json'))['outcome'])" | tr -d '\r')
for d in "$ROOT/pairs" "$ROOT/work"; do [ -e "$d" ] && say "deleted $d ($(bytes "$d") bytes)"; done
rm -rf "$ROOT/pairs" "$ROOT/work"
echo "ZZ AMEND7 READ DONE ZZ $out" | tee -a "$LOG"
