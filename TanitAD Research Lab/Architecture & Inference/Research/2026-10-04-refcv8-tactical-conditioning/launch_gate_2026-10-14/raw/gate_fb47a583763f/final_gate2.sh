#!/usr/bin/env bash
# refcv8 launch gate, end to end, for launch commit $1 whose binding records were produced on commit $2's tree
# (bind_full.sh; the trees differ ONLY outside both harnesses' code closures -- launch_gate.py + tests -- which the
# gate's own G-MAP / G-BOX closure check verifies against THIS launch tree). Waits for: the binding chain's
# ZZBINDFULL-OKZZ, the Thor suite's results/ALLDONE, and the dev-box G-SUITE-PINNED evidence. Then: suite verdict +
# suite-bind + the G-SUITE precheck (the gate's own judge), swap the staged tree into launch_tree, carry the binding
# records over (cp -p + PROVENANCE), and run the kit's `gate`. Markers: ZZFINAL-*ZZ; the gate's own ZZGATE-PASSZZ.
set -u
C=${1:?}; C12=${C:0:12}; B12=${2:?}; R=/home/nvidia/refcv8_run; G=/home/nvidia/gate_$C12; PY=/home/nvidia/venvs/tanitad-train/bin/python
DEVEV=$R/devbox_evidence_$C12; PIC=$R/pi_cost_approval.json
say() { echo "[final $(date +%H:%M:%S)] $*"; }
die() { say "REFUSED: $*"; echo "ZZFINAL-REFUSEDZZ"; exit 10; }
waitfor() { local what=$1 test=$2 i; for i in $(seq 1 720); do eval "$test" && { say "ready: $what"; return 0; }; sleep 30 200>&- 9>&-; done; die "timed out waiting for $what"; }
waitfor "binding chain" "grep -qE '^ZZBINDFULL-(OK|REFUSED)ZZ' $R/swap_and_bind_$B12.log 2>/dev/null"
grep -q '^ZZBINDFULL-OKZZ' $R/swap_and_bind_$B12.log || die "the binding chain on $B12 was REFUSED: $(tail -3 $R/swap_and_bind_$B12.log | tr '\n' ' ')"
waitfor "Thor suite" "[ -f $G/results/ALLDONE ]"
waitfor "dev-box evidence" "[ -s $DEVEV/G-SUITE-PINNED.json ]"
# ---- the suite verdict, bound, then judged by the gate's own G-SUITE function -----------------------------------
mkdir -p $G/v/res && { [ -e $G/v/res/results ] || ln -s $G/results $G/v/res/results; }
( cd $G && $PY gate_verdict.py $G/v > $G/v/verdict.out 2>&1 ) ; [ -s $G/v/verdict.json ] || die "no verdict.json"
( cd $G/CAND && PYTHONPATH=$G/CAND/stack $PY $G/CAND/stack/scripts/launch_gate.py suite-bind --verdict $G/v/verdict.json \
    --cand-tree $G/CAND --commit $C --out $G/suite_bound.json > $G/suite_bind.log 2>&1 )
grep -q "^ZZSUITE-BOUND-${C12}ZZ" $G/suite_bind.log && [ -s $G/suite_bound.json ] || die "suite-bind failed: $(tail -2 $G/suite_bind.log)"
# ---- swap: the bindings are finished, so nothing reads launch_tree any more ----------------------------------------
[ "$(cat $R/launch_tree_next/COMMIT)" = "$C" ] || die "launch_tree_next is not $C"
OLD=$(cat $R/launch_tree/COMMIT); [ ${#OLD} = 40 ] || die "launch_tree has no COMMIT"
[ -e $R/launch_tree_${OLD:0:12} ] && die "launch_tree_${OLD:0:12} exists"
c=$(ps -eo args | grep -c "[c]losure_run.py --binding"); [ "$c" = 0 ] || die "a binding run is still alive"
mv $R/launch_tree $R/launch_tree_${OLD:0:12} && mv $R/launch_tree_next $R/launch_tree || die "swap"
[ "$(cat $R/launch_tree/COMMIT)" = "$C" ] || die "launch_tree is not $C after the swap"
say "launch_tree is now $C (was $OLD)"
( cd $R/launch_tree && PYTHONPATH=$R/launch_tree/stack:$R/launch_tree/stack/scripts $PY $R/gsuite_precheck.py $R/launch_tree $G/suite_bound.json $C ) > $G/gsuite_precheck.txt 2>&1
grep -q "^G-SUITE PRECHECK: PASS" $G/gsuite_precheck.txt || die "G-SUITE precheck: $(grep -E 'OUTSIDE|reason|PRECHECK' $G/gsuite_precheck.txt | head -5 | tr '\n' ' ')"
say "G-SUITE precheck PASS"
# ---- the binding records, carried from bind_$B12 ------------------------------------------------------------------
BS=$R/bind_$B12; BD=$R/bind_$C12; [ -e $BD ] && die "$BD exists"
mkdir -p $BD/run_map $BD/run_box || die "mkdir"
cp -p $BS/run_map/g_map_overfit.json $BD/run_map/ && cp -p $BS/map_closure.json $BS/box_closure.json $BD/ && \
  cp -p $BS/run_box/gbo_binding.json $BD/run_box/ || die "copy records"
{ echo "binding records COPIED from $BS (run by the kit's bind on launch_tree = $B12; $C changes only launch_gate.py, tests,"
  echo "refcv8_prelaunch.sh and byte-identical copies of the box-audit INPUTS (md5 594c7119.../b291404c..., the files bind_full.sh"
  echo "prepared by hand) -- outside both harness code closures; G-MAP/G-BOX-OVERFIT verify the closure against THIS launch tree)"
  md5sum $BS/run_map/g_map_overfit.json $BS/map_closure.json $BS/run_box/gbo_binding.json $BS/box_closure.json \
         $BD/run_map/g_map_overfit.json $BD/map_closure.json $BD/run_box/gbo_binding.json $BD/box_closure.json; } > $BD/PROVENANCE.txt
H=$R/launch_tree/stack/scripts
for k in map box; do
  [ $k = map ] && REC=$BD/run_map/g_map_overfit.json || REC=$BD/run_box/gbo_binding.json
  $PY $H/refcv8_assemble_argv.py --assert-closure $BD/${k}_closure.json 2>&1 | grep -q '^ASSERT-OK' || die "$k closure"
  $PY $H/refcv8_assemble_argv.py --assert-record $REC --kind $k 2>&1 | grep -q '^ASSERT-OK' || die "$k record"
done
say "binding records asserted under $BD"
# ---- the gate (the kit: preflights, the binding asserts, launch_gate run --stage thor, the token asserted) -----------
echo "82c6295ae9bfb6b029570cfde7d10239  $PIC" | md5sum -c - > /dev/null 2>&1 || die "pi_cost_approval md5"
cd $R/launch_tree || die "cd"
bash $R/launch_tree/stack/scripts/refcv8_prelaunch.sh gate --code $R/launch_tree --commit $C --suite-verdict $G/suite_bound.json \
  --devbox-evidence $DEVEV --pi-cost-approval $PIC > $R/gate_$C12.out 2>&1 9>&- 200>&-
if grep -q '^ZZGATE-PASSZZ' $R/gate_$C12.out; then say "GATE PASS"; echo "ZZFINAL-GATE-PASSZZ"
else say "gate did not pass: $(tail -6 $R/gate_$C12.out | tr '\n' ' ')"; echo "ZZFINAL-GATE-NOTPASSZZ"; exit 11; fi
