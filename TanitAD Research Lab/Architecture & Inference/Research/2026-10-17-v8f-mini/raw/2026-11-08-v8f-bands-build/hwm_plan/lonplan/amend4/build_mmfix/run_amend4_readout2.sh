#!/bin/bash
# run_amend4_readout2.sh -- MM, 2026-11-17. Runs ONLY step 5 (the SCORE readout) of the AMEND_4 read.
# The resume run finished SCORE and readback for P / P23 / RS at both chassis (logs/score_s{0,1}.rc and logs/readback_s{0,1}.rc = 0),
# then the readout raised ModuleNotFoundError: lonplan_common at amend4_read.py:179. score() imported lonplan_common BEFORE
# _code_of_record_on_path() ran (the first fix placed the call before hwm_eval only). Fix 2: the helper is now the first line of
# score() and sha(); test_amend4.py 34 passed; a fresh process imports and hashes. Planning code and SCORE outputs are untouched.
# Last line of $ROOT/read.log:  ZZ AMEND4 READ DONE ZZ  |  ZZ AMEND4 READ FAIL ZZ <why>
source F:/TanitAD-scratch/actrsp/lonplan/amend4/build/env_amend4.sh
ROOT=F:/TanitAD-scratch/actrsp/lonplan/amend4
B=$ROOT/build
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
LOG=$ROOT/read.log
cd "$B" || exit 1
say() { echo "$(date -u +%T) $*" | tee -a "$LOG"; }
fail() { echo "ZZ AMEND4 READ FAIL ZZ $1" | tee -a "$LOG"; exit 1; }
say "READOUT-ONLY $(date -u +%FT%TZ) (MM fix 2 of the readout import; SCORE and readback kept)"
want=$(cut -c1-64 "$ROOT/AMEND_HWMPLAN_4.sha256"); got=$(sha256sum "$ROOT/AMEND_HWMPLAN_4.md" | cut -c1-64)
[ "$got" = "$want" ] || fail "AMEND_HWMPLAN_4.md sha256 $got is not the frozen $want"
for n in score readback; do for k in s0 s1; do [ "$(cat "$ROOT/logs/${n}_$k.rc" 2>/dev/null)" = "0" ] || fail "${n} $k rc is not 0"; done; done
for x in P P23 RS; do for k in s0 s1; do [ -f "$ROOT/score/${x}_$k/DONE" ] || fail "score/${x}_$k has no DONE"; done; done
$PY -u "$WC" 5.5 300 >> "$LOG" 2>&1 || fail "commit headroom stayed under 5.5 GB for 5 h (readout)"
$PY -u amend4_read.py score >> "$LOG" 2>&1 || fail "readout (see $LOG)"
echo "ZZ AMEND4 READ DONE ZZ" | tee -a "$LOG"
