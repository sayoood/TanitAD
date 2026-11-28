#!/usr/bin/env bash
# Waker for PREREG_V8F_DAGGER's box-2 chain (chain_v8f_dag.out). Exits on: the chain's end / stop marker (a BLOCKED, REFUSED or PARKED stop is a stop), a guard stop, the four gates passing (once; they come first after the banks),
# each bank pulled, or each run's pull landing on E:. Script polling, not model turns.
L=/c/Users/Admin/box2_ship/logs; O=$L/chain_v8f_dag.out
np0=$(grep -ac ' pulled to ' $O 2>/dev/null); np0=${np0:-0}
for i in $(seq 1 1500); do
  grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $O 2>/dev/null && { echo "CHAIN END"; tail -15 $O; exit 0; }
  grep -aq ZZGUARD-STOP $L/v8fdag_guard.out 2>/dev/null && { echo "GUARD: $(grep -a ZZGUARD-STOP $L/v8fdag_guard.out | tail -1)"; tail -8 $O; exit 4; }
  grep -aq '^ZZDAG-BLOCKED\|^ZZDAG-REFUSED' $O 2>/dev/null && { echo "BANK JUDGE: $(grep -a '^ZZDAG-' $O | tail -2)"; tail -6 $O; exit 0; }
  [ "$(grep -ac '^DAG gate: PASS' $O 2>/dev/null)" -ge 4 ] && [ ! -f $L/.dag_gates_seen ] && { touch $L/.dag_gates_seen; echo "GATES PASSED"; grep -a '^DAG gate' $O; tail -4 $O; exit 0; }
  n=$(grep -ac ' pulled to ' $O 2>/dev/null); [ "${n:-0}" -gt "$np0" ] && { echo "NEW PULL"; grep -a ' pulled to ' $O | tail -1; tail -6 $O; exit 0; }
  sleep 60
done
echo "WAKER TIMEOUT"; tail -5 $O
