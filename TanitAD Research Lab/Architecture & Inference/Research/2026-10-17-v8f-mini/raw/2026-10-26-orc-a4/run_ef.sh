#!/usr/bin/env bash
# PREREG_V8F_ORECOVER Amendment 4 (before any run): the ego probe's at-rest exemption. Fresh tree = archive of 7f06ff00 + the trainer edit, 3 tests, 3 mutants.
# No other job reads or mutates this tree. Sequential; each pytest run waits for < 2 pytest RUNS.
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
EF=E:/Projects/TanitAD-artifacts/ef
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
export PYTHONPATH="$EF/tree/stack;$EF/tree/taniteval;$EF/tree/stack/scripts"
mkdir -p $EF/p $EF/out
slot() { for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && return 0; sleep 20; done; return 1; }
cd $EF/tree
"$PY" "$PK/raw/2026-10-25-oacc/mutate_oacc.py" "$EF/tree" "$PY" --check > $EF/out/mutate_oacc_check.out 2>&1; echo "oacc anchors rc=$? $(tail -1 $EF/out/mutate_oacc_check.out)"
"$PY" "C:/Users/Admin/mixo_kill/tree/$PK/raw/2026-10-25-mixo/mutate_mixo.py" "$EF/tree" "$PY" --check > $EF/out/mutate_mixo_check.out 2>&1; echo "mixo anchors (the kill version) rc=$? $(tail -1 $EF/out/mutate_mixo_check.out)"
slot || { echo ZZEF-NO-SLOT; exit 3; }
cd $EF/tree/stack
"$PY" -m pytest -q -p no:cacheprovider --basetemp $EF/p/e tests/test_v8f_ego_perturb.py > $EF/out/pytest_ego_perturb.out 2>&1
echo "ego perturb tests rc=$? $(tail -1 $EF/out/pytest_ego_perturb.out)"
slot || { echo ZZEF-NO-SLOT; exit 3; }
"$PY" -m pytest -q -p no:cacheprovider --basetemp $EF/p/o tests/test_v8f_orc_chain.py tests/test_v8f_oacc.py > $EF/out/pytest_orc_chain_oacc.out 2>&1
echo "orc chain + oacc tests rc=$? $(tail -1 $EF/out/pytest_orc_chain_oacc.out)"
slot || { echo ZZEF-NO-SLOT; exit 3; }
cd $EF/tree
"$PY" "$PK/raw/2026-10-26-orecover/mutate_orecover.py" "$EF/tree" "$PY" > $EF/out/mutate_orecover.out 2>&1
echo "orecover proof rc=$? $(tail -2 $EF/out/mutate_orecover.out | tr '\n' ' ')"
git -C C:/Users/Admin/tanitad-push -c safe.directory='*' show 7f06ff00:stack/scripts/v8f_mini_train.py | diff - stack/scripts/v8f_mini_train.py > $EF/out/v8f_mini_train.py.vs7f06ff00.diff
echo "trainer: $(grep -c '^[<>]' $EF/out/v8f_mini_train.py.vs7f06ff00.diff) changed lines vs 7f06ff00"
echo ZZEF-DONE
