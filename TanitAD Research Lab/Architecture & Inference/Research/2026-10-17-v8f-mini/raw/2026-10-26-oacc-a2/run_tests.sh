#!/usr/bin/env bash
# OACC Amendment 2: the reader's test file and the two weight/lever mutants, each started only when fewer than 2 pytest RUNS are active.
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
X=E:/Projects/TanitAD-artifacts/oaccfix
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="$X/tree/stack;$X/tree/taniteval;$X/tree/stack/scripts"
slot() { for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && return 0; sleep 30; done; return 1; }
mkdir -p /e/Projects/TanitAD-artifacts/oaccfix/out
slot || { echo ZZOACCFIX-NO-SLOT; exit 3; }
cd /e/Projects/TanitAD-artifacts/oaccfix/tree/stack
echo "oacc tests start $(date '+%H:%M:%S')"
"$PY" -m pytest -q -p no:cacheprovider --basetemp $X/pt/s tests/test_v8f_oacc.py > $X/out/pytest_oacc.out 2>&1
echo "oacc tests rc=$? $(tail -1 $X/out/pytest_oacc.out)"
slot || { echo ZZOACCFIX-NO-SLOT; exit 3; }
cd /e/Projects/TanitAD-artifacts/oaccfix/tree
M="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-oacc/mutate_oacc.py"
"$PY" "$M" "$X/tree" "$PY" --check > $X/out/mutate_check.out 2>&1; echo "check rc=$? $(tail -1 $X/out/mutate_check.out)"
"$PY" "$M" "$X/tree" "$PY" --only "built levers" --tmp $X/pt/mut > $X/out/mutate_lever.out 2>&1; echo "mut levers rc=$? $(tail -2 $X/out/mutate_lever.out | tr '\n' ' ')"
"$PY" "$M" "$X/tree" "$PY" --only "loss weights" --tmp $X/pt/mut2 > $X/out/mutate_w.out 2>&1; echo "mut weights rc=$? $(tail -2 $X/out/mutate_w.out | tr '\n' ' ')"
echo ZZOACCFIX-DONE
