#!/bin/bash
# sweep.sh : the FULL-FORM regression sweep of the SAFEOBJ build, on the clean patched tree, at most TWO test processes at a time (binding resource cap).
#   usage: sweep.sh [PATCHED_TREE=C:/so/t] [BASE_TREE=C:/so/b]      (run it detached; it ends by writing TEST_TABLE.md and C:/so/logs_t_DONE)
# worker 1 and worker 2 each run their tests SEQUENTIALLY (run_tests.sh = one bounded `timeout 2400` python per test); then every failing test is rerun on the UNPATCHED tree (one at a time)
# and make_test_table.py judges PASS / PRE-EXISTING / REGRESSION from the two logs.  Kill, if ever needed, only by explicit PID.
T="${1:-C:/so/t}"; B="${2:-C:/so/b}"
HERE="$(cd "$(dirname "$0")" && pwd)"; RT="$HERE/run_tests.sh"; PKG="$(dirname "$HERE")"
rm -rf C:/so/logs_t1 C:/so/logs_t2 C:/so/logs_tb C:/so/logs_t_DONE
W1="refe/test_rl_s1b.py refe/test_rl_r0c.py refe/test_rl_epo.py refe/test_launch_gate.py refe/test_rls1_queue.py refe/test_s1aux_chain.py refe/test_p6_levers.py refe/test_rl_tlane_label.py refe/test_dump_scores.py refe/test_rl_terms_label.py refe/test_label_dir_refusal.py refe/test_flock_rw.py refe/test_gate_gpu_probes.py refe/test_p2r_chain.py refe/test_p6_overlap.py eval/test_aux_navtest_forward.py refe/test_aux_g12_bf16.py"
W2="refe/test_rl_safeobj.py refe/test_rls1_chain.py refe/test_aux_s1_wiring.py refe/test_p6_chain.py refe/test_gate_suite_prerun.py refe/test_p2rep_chain.py refe/test_s1c.py refe/test_p6_cpu_probe.py eval/test_rl_val_eval.py"
( while [ ! -f C:/so/logs_t_DONE ]; do echo "$(date +%H:%M:%S) python=$(tasklist 2>/dev/null | grep -ci '^python') total=$(tasklist 2>/dev/null | wc -l)" >> C:/so/logs_t_procs.txt; sleep 60; done ) &
MON=$!
bash "$RT" "$T" C:/so/logs_t1 $W1 &
P1=$!
bash "$RT" "$T" C:/so/logs_t2 $W2 &
P2=$!
wait $P1 $P2
FAILS=$(cat C:/so/logs_t1/SUMMARY.txt C:/so/logs_t2/SUMMARY.txt | grep -E ' rc=[1-9]' | sed 's/ rc=.*//' )
LIST=""
for f in $FAILS; do
  for p in refe/$f.py eval/$f.py; do [ -f "$B/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/$p" ] && LIST="$LIST $p"; done
done
[ -n "$LIST" ] && bash "$RT" "$B" C:/so/logs_tb $LIST
"${PYTHON:-C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe}" -X utf8 "$HERE/make_test_table.py" "$T" "$PKG" > "$PKG/logs/make_test_table.out" 2>&1
mkdir -p "$PKG/logs/sweep"; for d in logs_t1 logs_t2 logs_tb logs_b logs_b2; do mkdir -p "$PKG/logs/sweep/$d"; cp C:/so/$d/*.log C:/so/$d/SUMMARY.txt "$PKG/logs/sweep/$d/" 2>/dev/null; done
cp C:/so/logs_t_procs.txt "$PKG/logs/sweep/procs.txt" 2>/dev/null
date > C:/so/logs_t_DONE
