#!/bin/bash
# test_ki_marker_watch.sh <scratch dir> -- dev-box test (Git bash, no pod) of rh_ki_marker_watch.sh (RA's form) with the KIW_* path overrides pointing into the scratch dir.
# T1 the marker is ABSENT at the start: the watcher sets it at once with its own token (noclobber), records the token, and clears ONLY its own token at the chain's lock line (exit 0).
# T2 a FOREIGN marker is present at the start: nothing is set, nothing recorded, the foreign marker is left alone, exit 0 at once.
# T3 the marker is replaced by another owner before the lock line: NOT cleared (exit 0).
# T4 the chain ends before the lock line (`ZZ ROLLHIST-KI FAIL`): own token cleared, exit 1.
# T5 no chain process exists after the settle time: own token cleared, exit 2.
# T6 an empty marker present at the start (the Ruled session's old form) counts as foreign: not overwritten.
# T7 only chain-log lines written AFTER the watcher started count: an old lock line in the log does not clear the marker.
HERE=$(cd "$(dirname "$0")" && pwd)
W=$1; mkdir -p "$W"; ok=1
export KIW_SETTLE=1 KIW_POLL=1
t() { echo "$1 $2 $3"; [ "$2" = PASS ] || ok=0; }
# a fake chain PROCESS (the watcher checks that a process whose command line names rh_ki_chain.sh exists): a script of that name that only sleeps
mkdir -p "$W/fake"; printf '#!/bin/bash
sleep 120
' > "$W/fake/rh_ki_chain.sh"
startfake() { bash "$W/fake/rh_ki_chain.sh" & FP=$!; }
stopfake() { kill $FP 2>/dev/null; wait $FP 2>/dev/null; }
setup() { rm -rf "$W/$1"; mkdir -p "$W/$1"; export KIW_MARKER="$W/$1/N1DW_READY" KIW_MINE="$W/$1/mine" KIW_CHAINLOG="$W/$1/chain.log" KIW_LOG="$W/$1/watch.log"; : > $KIW_CHAINLOG; }
# ---- T1
setup t1; startfake
bash $HERE/rh_ki_marker_watch.sh & P=$!
sleep 3
[ -s $KIW_MARKER ] && [ "$(cat $KIW_MARKER)" = "$(cat $KIW_MINE)" ] && grep -q "^KI-" $KIW_MARKER && t T1a PASS "own token set at the start, token recorded" || t T1a FAIL "no own marker"
echo "2026 chain start" >> $KIW_CHAINLOG; sleep 1
[ -s $KIW_MARKER ] && t T1b PASS "marker kept while the chain waits for the lock" || t T1b FAIL "marker lost"
echo "2026 GPU queue lock held to the end of the chain" >> $KIW_CHAINLOG
wait $P; rc=$?; stopfake
[ $rc = 0 ] && [ ! -e $KIW_MARKER ] && [ ! -e $KIW_MINE ] && t T1c PASS "cleared at the lock line, exit 0" || t T1c FAIL "rc=$rc"
# ---- T2
setup t2; echo foreign > $KIW_MARKER; startfake
bash $HERE/rh_ki_marker_watch.sh & P=$!
wait $P; rc=$?; stopfake
[ $rc = 0 ] && [ "$(cat $KIW_MARKER)" = foreign ] && [ ! -e $KIW_MINE ] && grep -q "present and not ours" $KIW_LOG && t T2 PASS "foreign marker not overwritten, nothing recorded, exit 0, logged" || t T2 FAIL "rc=$rc marker=$(cat $KIW_MARKER 2>/dev/null)"
# ---- T3
setup t3; startfake
bash $HERE/rh_ki_marker_watch.sh & P=$!
sleep 3
[ -s $KIW_MARKER ] && t T3a PASS "set" || t T3a FAIL "not set"
echo "another-owner" > $KIW_MARKER
echo "GPU queue lock held to the end of the chain" >> $KIW_CHAINLOG; wait $P; rc=$?; stopfake
[ $rc = 0 ] && [ "$(cat $KIW_MARKER)" = another-owner ] && t T3b PASS "a marker with another content is NOT cleared" || t T3b FAIL "rc=$rc marker=$(cat $KIW_MARKER 2>/dev/null)"
# ---- T4
setup t4; startfake
bash $HERE/rh_ki_marker_watch.sh & P=$!
sleep 3
echo "ZZ ROLLHIST-KI FAIL ZZ step gsuite" >> $KIW_CHAINLOG; wait $P; rc=$?; stopfake
[ $rc = 1 ] && [ ! -e $KIW_MARKER ] && [ ! -e $KIW_MINE ] && t T4 PASS "chain ended before the lock: cleared, exit 1" || t T4 FAIL "rc=$rc"
# ---- T5 (no chain process)
setup t5
bash $HERE/rh_ki_marker_watch.sh & P=$!
wait $P; rc=$?
[ $rc = 2 ] && [ ! -e $KIW_MARKER ] && [ ! -e $KIW_MINE ] && t T5 PASS "no chain process: own token cleared, exit 2" || t T5 FAIL "rc=$rc"
# ---- T6 (an empty marker is foreign)
setup t6; : > $KIW_MARKER; startfake
bash $HERE/rh_ki_marker_watch.sh & P=$!
wait $P; rc=$?; stopfake
[ $rc = 0 ] && [ -e $KIW_MARKER ] && [ ! -s $KIW_MARKER ] && [ ! -e $KIW_MINE ] && t T6 PASS "an empty marker is left alone, exit 0" || t T6 FAIL "rc=$rc"
# ---- T7 (an OLD lock line in the log does not count)
setup t7; echo "2025 GPU queue lock held to the end of the chain" >> $KIW_CHAINLOG; startfake
bash $HERE/rh_ki_marker_watch.sh & P=$!
sleep 4
kill -0 $P 2>/dev/null && [ -s $KIW_MARKER ] && t T7a PASS "the old lock line did not clear the marker" || t T7a FAIL "cleared by an old line"
echo "GPU queue lock held to the end of the chain" >> $KIW_CHAINLOG; wait $P; rc=$?; stopfake
[ $rc = 0 ] && [ ! -e $KIW_MARKER ] && t T7b PASS "cleared by the new lock line" || t T7b FAIL "rc=$rc"
echo "ZZ KI-MARKER-WATCH TEST $([ $ok = 1 ] && echo PASS || echo FAIL)"
[ $ok = 1 ]
