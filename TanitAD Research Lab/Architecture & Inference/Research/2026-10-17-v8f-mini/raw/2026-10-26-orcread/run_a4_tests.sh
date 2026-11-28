#!/usr/bin/env bash
# The reader's suite and mutation proof after Amendment 4, each started only when fewer than 2 pytest RUNS are active (the session cap).
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd /c/Users/Admin/orcread
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
BT=E:/Projects/TanitAD-artifacts/orcread_build/pt
mkdir -p /e/Projects/TanitAD-artifacts/orcread_build/pt
slot() { for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && return 0; sleep 30; done; return 1; }
slot || { echo "ZZA4-NO-SLOT suite"; exit 3; }
echo "suite start $(date '+%H:%M:%S')"
"$PY" -m pytest -q -p no:cacheprovider --basetemp $BT/a4_suite test_orc_read.py > out/pytest_full_a4.out 2>&1
echo "suite rc=$? $(tail -1 out/pytest_full_a4.out)"
slot || { echo "ZZA4-NO-SLOT mutation"; exit 3; }
echo "mutation start $(date '+%H:%M:%S')"
"$PY" mutate_orc.py --out out/mutation_a4.out > out/mutation_a4.log 2>&1
echo "mutation rc=$? $(grep -ciE 'KILLED' out/mutation_a4.out) killed; $(grep -ciE 'SURVIV' out/mutation_a4.out) survived lines"
echo ZZA4-DONE
cd /c/Users/Admin/orcread && sha256sum -c snap_a4_SHA256.txt > out/snap_a4_check.out 2>&1; echo "sources vs snapshot rc=$?"
