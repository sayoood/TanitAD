#!/bin/bash
# test_rk_marker_watch.sh <scratch dir> -- dev-box test (Git bash, no pod) of rh_rk_marker_watch.sh with the RKW_* path overrides pointing into the scratch dir.
# T1 the take-over: the ADD123 chain holds an EMPTY marker; the watcher must NOT touch it while ADD123 runs, must set its own token within a few seconds of ADD123's `marker removed (mine)` line, and clear ONLY its own token at the chain's lock line (exit 0).
# T2 a foreign marker present at the end: never overwritten; set as soon as it is removed.
# T3 the marker replaced by another owner before the lock line: NOT cleared.
# T4 the chain ends before the lock line (`ZZ ROLLHIST-RK FAIL`): own token cleared, exit 1.
# T5 the end is seen only through the last line `ZZ ADD123 DONE ZZ` (no `marker removed` line): same take-over.
HERE=$(cd "$(dirname "$0")" && pwd)
W=$1; mkdir -p "$W"; ok=1
export RKW_SETTLE=1
t() { echo "$1 $2 $3"; [ "$2" = PASS ] || ok=0; }
# a fake chain PROCESS (phase 3 checks that a process whose command line names rh_rk_chain.sh exists): a script of that name that only sleeps
mkdir -p "$W/fake"; printf '#!/bin/bash
sleep 120
' > "$W/fake/rh_rk_chain.sh"
startfake() { bash "$W/fake/rh_rk_chain.sh" & FP=$!; }
stopfake() { kill $FP 2>/dev/null; wait $FP 2>/dev/null; }
setup() { rm -rf "$W/$1"; mkdir -p "$W/$1"; export RKW_MARKER="$W/$1/N1DW_READY" RKW_MINE="$W/$1/mine" RKW_ADDLOG="$W/$1/add.log" RKW_CHAINLOG="$W/$1/chain.log" RKW_LOG="$W/$1/watch.log"; : > $RKW_CHAINLOG; echo "x marker set; waiting" > $RKW_ADDLOG; }
# ---- T1
setup t1; touch $RKW_MARKER; : > "$W/t1/add_mine"
startfake
bash $HERE/rh_rk_marker_watch.sh & P=$!
sleep 5
[ -e $RKW_MARKER ] && [ ! -s $RKW_MARKER ] && [ ! -e $RKW_MINE ] && t T1a PASS "the watcher left the ADD123 marker alone while ADD123 ran" || t T1a FAIL "marker touched early"
rm -f $RKW_MARKER; echo "t marker removed (mine)" >> $RKW_ADDLOG
sleep 5
[ -s $RKW_MARKER ] && [ "$(cat $RKW_MARKER)" = "$(cat $RKW_MINE)" ] && grep -q "^RK-" $RKW_MARKER && t T1b PASS "own token set after ADD123's marker removal" || t T1b FAIL "no own marker"
echo "ZZ ADD123 DONE ZZ" >> $RKW_ADDLOG; sleep 1
[ -s $RKW_MARKER ] && t T1c PASS "marker still there after the last ADD123 line (no double set / clear)" || t T1c FAIL "marker lost"
echo "2026 GPU queue lock held to the end of the chain" >> $RKW_CHAINLOG
wait $P; rc=$?; stopfake
[ $rc = 0 ] && [ ! -e $RKW_MARKER ] && [ ! -e $RKW_MINE ] && t T1d PASS "cleared at the lock line, exit 0" || t T1d FAIL "rc=$rc"
# ---- T2
setup t2; echo foreign > $RKW_MARKER; echo "ZZ ADD123 DONE ZZ" >> $RKW_ADDLOG
startfake
bash $HERE/rh_rk_marker_watch.sh & P=$!
sleep 6
[ "$(cat $RKW_MARKER)" = foreign ] && [ ! -e $RKW_MINE ] && t T2a PASS "foreign marker not overwritten" || t T2a FAIL "overwritten"
rm -f $RKW_MARKER; sleep 4
[ -s $RKW_MARKER ] && grep -q "^RK-" $RKW_MARKER && t T2b PASS "set once the foreign marker was gone" || t T2b FAIL "not set"
echo "GPU queue lock held to the end of the chain" >> $RKW_CHAINLOG; wait $P; rc=$?; stopfake
[ $rc = 0 ] && [ ! -e $RKW_MARKER ] && t T2c PASS "cleared, exit 0" || t T2c FAIL "rc=$rc"
# ---- T3
setup t3; echo "marker removed (mine)" >> $RKW_ADDLOG
startfake
bash $HERE/rh_rk_marker_watch.sh & P=$!
sleep 5
[ -s $RKW_MARKER ] && t T3a PASS "set" || t T3a FAIL "not set"
echo "another-owner" > $RKW_MARKER
echo "GPU queue lock held to the end of the chain" >> $RKW_CHAINLOG; wait $P; rc=$?; stopfake
[ $rc = 0 ] && [ "$(cat $RKW_MARKER)" = another-owner ] && t T3b PASS "a marker with another content is NOT cleared" || t T3b FAIL "rc=$rc marker=$(cat $RKW_MARKER 2>/dev/null)"
# ---- T4
setup t4; echo "ZZ ADD123 FAIL ZZ smoke rc=1" >> $RKW_ADDLOG
startfake
bash $HERE/rh_rk_marker_watch.sh & P=$!
sleep 4
echo "ZZ ROLLHIST-RK FAIL ZZ step gsuite" >> $RKW_CHAINLOG; wait $P; rc=$?; stopfake
[ $rc = 1 ] && [ ! -e $RKW_MARKER ] && t T4 PASS "chain ended before the lock: cleared, exit 1" || t T4 FAIL "rc=$rc"
# ---- T5 (last line only)
setup t5
startfake
bash $HERE/rh_rk_marker_watch.sh & P=$!
sleep 3; echo "ZZ ADD123 STOP ZZ x" >> $RKW_ADDLOG; sleep 4
[ -s $RKW_MARKER ] && t T5 PASS "set on the last line alone" || t T5 FAIL "not set"
echo "GPU queue lock held to the end of the chain" >> $RKW_CHAINLOG; wait $P; stopfake
echo "ZZ RK-MARKER-WATCH TEST $([ $ok = 1 ] && echo PASS || echo FAIL)"
[ $ok = 1 ]
