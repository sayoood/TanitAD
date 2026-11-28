#!/usr/bin/env bash
# One sequential runner (no two waiters race for a pytest slot). In order:
#  1. the O-RECOVER hold's verification tree (tip 04d56671 + the hold overlaid): test_v8f_orc_chain.py, the reader suite at A3, and the queue
#     tests of all 14 files that read the queue (incl. the five widened ones);
#  2. the MIXO re-check's coupled label-pin check (run_mx.sh's never ran: pytest --basetemp needs its parent to exist);
#  3. the reader suite in its build dir after apply_a3b.py (the evidence the hold banks).
# Each pytest run starts only when fewer than 2 pytest RUNS are active (the session cap).
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
OC=E:/Projects/TanitAD-artifacts/oc; MX=E:/Projects/TanitAD-artifacts/mx
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
mkdir -p $OC/p $OC/out $MX/t E:/Projects/TanitAD-artifacts/orcread_build/pt
slot() { for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && return 0; sleep 20; done; return 1; }
export PYTHONPATH="$OC/tree/stack;$OC/tree/taniteval;$OC/tree/stack/scripts"
cd $OC/tree/stack
slot || { echo ZZOC-NO-SLOT; exit 3; }
"$PY" -m pytest -q -p no:cacheprovider --basetemp $OC/p/c tests/test_v8f_orc_chain.py > $OC/out/orc_chain_tests.out 2>&1
echo "orc chain tests rc=$? $(tail -1 $OC/out/orc_chain_tests.out)"
slot || { echo ZZOC-NO-SLOT; exit 3; }
(cd "$OC/tree/$PK/raw/2026-10-26-orcread" && "$PY" -m pytest -q -p no:cacheprovider --basetemp $OC/p/r test_orc_read.py > $OC/out/orc_reader_tests.out 2>&1)
echo "orc reader tests (tree) rc=$? $(tail -1 $OC/out/orc_reader_tests.out)"
slot || { echo ZZOC-NO-SLOT; exit 3; }
"$PY" -m pytest -q -p no:cacheprovider --basetemp $OC/p/q -k queue \
  tests/test_v8f_cz.py tests/test_v8f_dscale.py tests/test_v8f_flat.py tests/test_v8f_grid.py tests/test_v8f_hist.py tests/test_v8f_kind.py tests/test_v8f_kindx_s1.py tests/test_v8f_lora_ar2.py \
  tests/test_v8f_mini_full.py tests/test_v8f_mixo.py tests/test_v8f_oacc.py tests/test_v8f_pgt.py tests/test_v8f_tac.py tests/test_v8f_wc.py > $OC/out/queue_tests.out 2>&1
echo "queue tests (14 files) rc=$? $(tail -1 $OC/out/queue_tests.out)"
echo ZZOC-ORC-DONE
export PYTHONPATH="$MX/tree/stack;$MX/tree/taniteval;$MX/tree/stack/scripts"
slot || { echo ZZOC-NO-SLOT; exit 3; }
cd $MX/tree && "$PY" -m pytest -q -p no:cacheprovider --basetemp $MX/t/c2 stack/tests/test_v8f_flat.py stack/tests/test_v8f_oacc.py -k every_existing_label_is_byte_identical > $MX/out/coupled_check.out 2>&1
echo "mixo coupled rc=$? $(tail -1 $MX/out/coupled_check.out)"
slot || { echo ZZOC-NO-SLOT; exit 3; }
cd /c/Users/Admin/orcread && "$PY" -m pytest -q -p no:cacheprovider --basetemp E:/Projects/TanitAD-artifacts/orcread_build/pt/a3b test_orc_read.py > out/pytest_full_a3b.out 2>&1
echo "orc reader suite (build dir) rc=$? $(tail -1 out/pytest_full_a3b.out)"
echo ZZOC-DONE
