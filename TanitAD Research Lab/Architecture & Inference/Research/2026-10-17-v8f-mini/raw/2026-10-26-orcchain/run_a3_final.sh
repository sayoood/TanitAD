#!/usr/bin/env bash
# Amendment 3 final pipeline: (1) the whole new test file, (2) the 14-file queue-structure suite, (3) the FULL mutation proof (with its unmutated control); each run starts only when pytest_runs.py prints 0 or 1
B=/e/Projects/TanitAD-artifacts/orcchain_build
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
O=/c/Users/Admin/orcchain/out
waitcap() { while true; do n=$($PY C:/Users/Admin/orcread/pytest_runs.py 2>/dev/null | tr -d '\r\n '); if [ "$n" = "0" ] || [ "$n" = "1" ]; then return; fi; sleep 4; done; }
waitcap; bash $B/run_orc_tests.sh full3; cp $B/out/orc_tests_full3.out $O/orc_tests_full3.out
waitcap; bash $B/run_queue_tests.sh $B/tree final3; cp $B/out/queue_tests_final3.out $O/queue_tests_final3.out
waitcap; $PY C:/Users/Admin/orcchain/tools/mutate_orcchain.py --tree $B/tree --out $O/mutation.out > $B/out/mutation_run3.log 2>&1
echo "mutation rc=$?" >> $B/out/mutation_run3.log
cp $B/out/mutation_run3.log $O/mutation_run.log
echo done > $B/out/a3_done.txt
