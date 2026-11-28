#!/bin/bash
# run_amend4_resume.sh -- MM, 2026-11-17. Resumes the AMEND_4 read after its first run stopped at the TRAIN readout:
#   `amend4_read.py train` raised ModuleNotFoundError: hwm_eval (it never put the code of record on sys.path; amend4_run.py
#   does that through lonplan_common). Fixed in amend4_read.py (_code_of_record_on_path, lazy); test_amend4.py 34 passed after the fix;
#   `amend4_read.py train` then wrote TRAIN_STAGE_s{0,1}.json with C-RS ok at both chassis (RS - A0 +0.3545 / +0.3385).
# Steps 0-2 are NOT repeated: the frozen-spec check and tests passed, controls are GREEN (controls/CONTROLS.json, 09:54),
# and the TRAIN plans are complete (logs/train_s{0,1}.rc = 0). Planning code is untouched by the fix. This runs steps 3-5 of run_amend4.sh unchanged.
# Last line of $ROOT/read.log:  ZZ AMEND4 READ DONE ZZ  |  ZZ AMEND4 READ FAIL ZZ <why>
source F:/TanitAD-scratch/actrsp/lonplan/amend4/build/env_amend4.sh
ROOT=F:/TanitAD-scratch/actrsp/lonplan/amend4
B=$ROOT/build
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$ROOT/read.log
cd "$B" || exit 1
say() { echo "$(date -u +%T) $*" | tee -a "$LOG"; }
fail() { echo "ZZ AMEND4 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
gate() { $PY -u "$WC" 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h ($1)"; }
say "RESUME $(date -u +%FT%TZ) (MM fix of the readout import; steps 0-2 kept)"
want=$(cut -c1-64 "$ROOT/AMEND_HWMPLAN_4.sha256"); got=$(sha256sum "$ROOT/AMEND_HWMPLAN_4.md" | cut -c1-64)
[ "$got" = "$want" ] || fail "AMEND_HWMPLAN_4.md sha256 $got is not the frozen $want"
grep -q '"ok": true' "$ROOT/controls/CONTROLS.json" || fail "controls/CONTROLS.json is not ok"
for k in s0 s1; do [ "$(cat "$ROOT/logs/train_$k.rc")" = "0" ] || fail "train $k rc is not 0"; [ -f "$ROOT/readout/TRAIN_STAGE_$k.json" ] || fail "TRAIN_STAGE_$k.json missing"; done
for x in P P23 RS; do for k in s0 s1; do [ -n "$(ls -A "$ROOT/score/${x}_$k" 2>/dev/null)" ] && fail "score/${x}_$k is not empty"; done; done
say "TRAIN stage done, C-RS sign holds at both chassis"
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
scr() { for x in P P23 RS; do $PY -u amend4_stage.py plan --stage score --arm $x --chassis "$1" || return 1; done; }
pair score scr
for k in s0 s1; do for x in P P23 RS; do cmp -s "$ROOT/score/${x}_$k/windows.npz" "F:/TanitAD-scratch/actrsp/lonplan/score/A0_$k/windows.npz" || fail "score/${x}_$k windows differ from A0_$k"; done; done
rbk() { for x in P P23 RS; do $PY -u amend4_run.py readback --out "$ROOT/score/${x}_$1" || return 1; done; }
pair readback rbk
gate readout
$PY -u amend4_read.py score >> "$LOG" 2>&1 || fail "readout (see $LOG)"
echo "ZZ AMEND4 READ DONE ZZ" | tee -a "$LOG"
