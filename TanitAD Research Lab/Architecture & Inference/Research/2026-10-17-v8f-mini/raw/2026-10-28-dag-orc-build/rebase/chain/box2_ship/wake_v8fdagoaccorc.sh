#!/usr/bin/env bash
# Waker for PREREG_V8F_DAG_ORC's box-2 chain (chain_v8f_dagoaccorc.out). Exits on: the chain's end / stop marker, a guard stop, a named stop (K-REACH, disk floor), the four A / REP gates passing (once; they come first; counted by their TOKEN lines -- the asserter prints `DAG-ORC (DAG-OACC base) gate: PASS` twice per gate, once in `check` and once in `token`),
# or each run's pull landing on E:. Script polling, not model turns.
L=/c/Users/Admin/box2_ship/logs; O=$L/chain_v8f_dagoaccorc.out
np0=$(grep -ac ' pulled to ' $O 2>/dev/null); np0=${np0:-0}
for i in $(seq 1 1500); do
  grep -aq '^ZZV8FMINI-STOP\|^ZZV8FMINI-DONE' $O 2>/dev/null && { echo "CHAIN END"; tail -15 $O; exit 0; }
  grep -aq ZZGUARD-STOP $L/v8fdagoaccorc_guard.out 2>/dev/null && { echo "GUARD: $(grep -a ZZGUARD-STOP $L/v8fdagoaccorc_guard.out | tail -1)"; tail -8 $O; exit 4; }
  grep -aq '^ZZDAGOACCORC-CONTROL-FAIL\|^ZZDISK-FLOOR-STOP' $O 2>/dev/null && { echo "NAMED STOP: $(grep -a '^ZZDAGOACCORC-\|^ZZDISK-' $O | tail -2)"; tail -6 $O; exit 0; }
  [ "$(grep -ac '^gate token written for' $O 2>/dev/null)" -ge 4 ] && [ ! -f $L/.dagoaccorc_gates_seen ] && { touch $L/.dagoaccorc_gates_seen; echo "GATES PASSED"; grep -a '^DAG-ORC (DAG-OACC base) gate' $O; tail -4 $O; exit 0; }
  n=$(grep -ac ' pulled to ' $O 2>/dev/null); [ "${n:-0}" -gt "$np0" ] && { echo "NEW PULL"; grep -a ' pulled to ' $O | tail -1; tail -6 $O; exit 0; }
  sleep 60
done
echo "WAKER TIMEOUT"; tail -5 $O
