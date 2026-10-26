#!/usr/bin/env bash
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
X=E:/Projects/TanitAD-artifacts/oaccfix; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="$X/tree/stack;$X/tree/taniteval;$X/tree/stack/scripts"
M="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-oacc/mutate_oacc.py"
for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && break; sleep 30; done
cd $X/tree && "$PY" "$M" "$X/tree" "$PY" --idx "$1" --tmp $X/m > $X/out/mutate_oacc_sel.out 2>&1
echo "rc=$? $(grep -acE 'RED' $X/out/mutate_oacc_sel.out) RED-lines, $(grep -aciE 'SURVIV|GREEN ' $X/out/mutate_oacc_sel.out) survivor-lines"; tail -3 $X/out/mutate_oacc_sel.out
