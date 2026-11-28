#!/usr/bin/env bash
# MM 2026-10-28: BASELINE for run_label_tests.sh -- the same overlay recipe (tip 6cd741b6: stack + taniteval + the v8f-mini code dir + the eval grid),
# UNPATCHED queue + UNEDITED tests. Its failure set is the environment's; the label edit must add nothing to it. Marker ZZBASE-DONE.
S=F:/TanitAD-scratch/dagorc_rebase_tmp; O=$S/overlay_base
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
T=6cd741b63b3b629282d48f3b88d86f99264f9c9f
PYV=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
L=/e/Projects/TanitAD-artifacts/dagorc_rebase/chain/logs/label_tests_baseline.out
export CUDA_VISIBLE_DEVICES=-1
for i in $(seq 1 120); do n=$($PYV C:/Users/Admin/orcread/pytest_runs.py 2>/dev/null | tail -1); [ "${n:-9}" -lt 2 ] && break; sleep 30; done
mkdir -p "$O" && [ -z "$(ls -A "$O")" ] || { echo "ZZBASE-STOP overlay dir not empty"; exit 1; }
git -C C:/Users/Admin/tanitad-push -c safe.directory='*' -c core.autocrlf=false archive --format=tar $T stack taniteval "$PK/code" "$PK/raw/eval_grid_windows.json" | tar -xf - -C "$O" || { echo "ZZBASE-STOP archive"; exit 1; }
TESTS="tests/test_v8f_flat.py tests/test_v8f_oacc.py tests/test_v8f_kind.py tests/test_v8f_mixo.py tests/test_v8f_lora_ar2.py tests/test_v8f_orc_chain.py"
(cd "$O/stack" && $PYV -m pytest -q -p no:cacheprovider --basetemp=$S/pt_base/base $TESTS 2>&1 | tail -40) > $L
tail -3 $L; echo ZZBASE-DONE
