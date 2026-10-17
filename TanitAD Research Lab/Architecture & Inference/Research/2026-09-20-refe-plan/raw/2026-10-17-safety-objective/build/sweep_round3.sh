#!/bin/bash
# sweep_round3.sh [TREE=C:/so/t6] : the FULL-FORM reruns of round 3 on the round-3 tree, at most TWO test processes at a time (binding resource cap; every test under `timeout 2400`, see run_tests.sh).
# The tests are the ones that NAME a file round 3 edited (grep over refe/test_*.py eval/test_*.py):
#   eval/safeobj_read.py   : test_safeobj_read, test_safeobj_gate (new)
#   refe/rls1_checks.py    : test_rl_safeobj, test_rl_tlane_label, test_rls1_chain, test_rls1_queue, test_s1aux_chain, test_safeobj_read, test_safeobj_gate
#   refe/test_rl_safeobj.py: test_launch_gate, test_rl_r0c, test_rl_s1b, test_rls1_chain      refe/test_rls1_queue.py: test_rls1_chain
# test_launch_gate runs through run_tests_sc.sh (sitecustomize bash shim in EVERY python process: a bare `bash` resolves to the WSL launcher on this box, BUILD.md section 3).
# Worker 2 starts 180 s after worker 1: the queue test's first scenarios are timing-sensitive (a dev run that started beside test_rl_safeobj failed at "returned before S1 was over", a race of the fake
# S1 chain's start under load, in code round 3 did not touch; it passed alone).  Run it detached; it ends by writing C:/so/logs_r3_DONE.   Kill, if ever needed, only by explicit PID.
T="${1:-C:/so/t6}"
HERE="$(cd "$(dirname "$0")" && pwd)"; RT="$HERE/run_tests.sh"; RTS="$HERE/run_tests_sc.sh"
rm -rf C:/so/logs_r3_1 C:/so/logs_r3_2 C:/so/logs_r3_DONE C:/so/logs_r3_procs.txt
( while [ ! -f C:/so/logs_r3_DONE ]; do echo "$(date +%H:%M:%S) python=$(tasklist 2>/dev/null | grep -ci '^python') total=$(tasklist 2>/dev/null | wc -l)" >> C:/so/logs_r3_procs.txt; sleep 60; done ) &
bash "$RT" "$T" C:/so/logs_r3_1 refe/test_rls1_queue.py refe/test_rl_tlane_label.py refe/test_rl_r0c.py refe/test_rl_s1b.py &
P1=$!
( sleep 180; bash "$RTS" "$T" C:/so/logs_r3_2 refe/test_launch_gate.py; bash "$RT" "$T" C:/so/logs_r3_2 refe/test_rl_safeobj.py refe/test_rls1_chain.py refe/test_s1aux_chain.py eval/test_safeobj_read.py eval/test_safeobj_gate.py ) &
P2=$!
wait $P1 $P2
date > C:/so/logs_r3_DONE
