#!/usr/bin/env bash
# After R77 (a mutant landed in 04d56671): verify the LANDED tip itself, then re-run the void MIXO trainer re-check on the correct trainer.
#  1. archive of the tip (short path) -> the test files of everything landed today that touches the trainer, the queue, OACC, O-RECOVER, MIXO.
#  2. mx tree: trainer := the tip's blob (the copy at ~11:35 carried an OACC mutant at line 1216); --check, the 15 trainer mutants, the coupled check.
# One process, sequential; every pytest run waits for < 2 pytest RUNS.
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
TIP=$1
TV=E:/Projects/TanitAD-artifacts/tv; MX=E:/Projects/TanitAD-artifacts/mx
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
G="git -C C:/Users/Admin/tanitad-push -c safe.directory=* -c core.autocrlf=false"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
mkdir -p $TV/out $TV/p $MX/t $MX/out
slot() { for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && return 0; sleep 20; done; return 1; }
[ -d $TV/tree ] && mv $TV/tree $TV/tree_old_$$
mkdir -p $TV/tree
$G archive $TIP -- stack taniteval "$PK" "Project Steering/PREREG_V8F_MIXO.md" "Project Steering/PREREG_V8F_OACC.md" "Project Steering/PREREG_V8F_FLAT.md" "Project Steering/PREREG_V8F_ORECOVER.md" | tar -x -C $TV/tree || { echo ZZTV-ARCHIVE-FAILED; exit 2; }
echo "tip $TIP archived; trainer line 745: $(sed -n 745p $TV/tree/stack/scripts/v8f_mini_train.py | tr -s ' ')"
export PYTHONPATH="$TV/tree/stack;$TV/tree/taniteval;$TV/tree/stack/scripts"
cd $TV/tree/stack
slot || { echo ZZTV-NO-SLOT; exit 3; }
"$PY" -m pytest -q -p no:cacheprovider --basetemp $TV/p/a tests/test_v8f_oacc.py tests/test_v8f_ego_perturb.py tests/test_v8f_orc_chain.py tests/test_v8f_mixo.py \
  tests/test_v8f_flat.py tests/test_v8f_kind.py tests/test_v8f_lora_ar2.py tests/test_v8f_kindx_s1.py > $TV/out/tip_tests.out 2>&1
echo "tip tests rc=$? $(tail -1 $TV/out/tip_tests.out)"
slot || { echo ZZTV-NO-SLOT; exit 3; }
"$PY" -m pytest -q -p no:cacheprovider --basetemp $TV/p/q -k queue \
  tests/test_v8f_cz.py tests/test_v8f_dscale.py tests/test_v8f_grid.py tests/test_v8f_hist.py tests/test_v8f_mini_full.py tests/test_v8f_pgt.py tests/test_v8f_tac.py tests/test_v8f_wc.py > $TV/out/tip_queue_rest.out 2>&1
echo "tip queue tests (the other 8 files) rc=$? $(tail -1 $TV/out/tip_queue_rest.out)"
echo ZZTV-TIP-DONE
# ---- 2. the MIXO trainer re-check, on the correct trainer
$G show "$TIP:stack/scripts/v8f_mini_train.py" > $MX/tree/stack/scripts/v8f_mini_train.py
[ "$(git hash-object --no-filters $MX/tree/stack/scripts/v8f_mini_train.py)" = "$($G rev-parse $TIP:stack/scripts/v8f_mini_train.py)" ] || { echo ZZTV-MX-TRAINER-NOT-TIP; exit 4; }
for f in stack/tests/test_v8f_oacc.py "$PK/code/v8f_oacc_read.py" "$PK/raw/2026-10-25-oacc/mutate_oacc.py"; do $G show "$TIP:$f" > "$MX/tree/$f"; done
mkdir -p $MX/out/void_r77 && mv $MX/out/mutate_mixo_train.out $MX/out/mutate_mixo_check.out $MX/out/coupled_check.out $MX/out/tree_files.out $MX/out/void_r77/ 2>/dev/null
export PYTHONPATH="$MX/tree/stack;$MX/tree/taniteval;$MX/tree/stack/scripts" MIXO_MUT_BASE=$MX/t2 TMP=$MX/st TEMP=$MX/st
mkdir -p $MX/t2 $MX/st
cd $MX/tree
for f in stack/scripts/v8f_mini_train.py stack/tests/test_v8f_mixo.py "$PK/raw/2026-10-25-mixo/mutate_mixo.py"; do echo "$f: $([ "$($G rev-parse "$TIP:$f")" = "$(git hash-object --no-filters "$f")" ] && echo 'tip bytes' || echo 'changed vs tip')"; done > $MX/out/tree_files.out
M="$PK/raw/2026-10-25-mixo/mutate_mixo.py"
"$PY" "$M" "$MX/tree" "$PY" --check > $MX/out/mutate_mixo_check.out 2>&1; echo "mixo check rc=$? $(tail -1 $MX/out/mutate_mixo_check.out)"
slot || { echo ZZTV-NO-SLOT; exit 3; }
"$PY" -m pytest -q -p no:cacheprovider --basetemp $MX/t/c3 stack/tests/test_v8f_flat.py stack/tests/test_v8f_oacc.py -k every_existing_label_is_byte_identical > $MX/out/coupled_check.out 2>&1
echo "mixo coupled rc=$? $(tail -1 $MX/out/coupled_check.out)"
slot || { echo ZZTV-NO-SLOT; exit 3; }
"$PY" "$M" "$MX/tree" "$PY" --only v8f_mini_train.py > $MX/out/mutate_mixo_train.out 2>&1
echo "mixo trainer mutants rc=$? $(grep -ac '^RED' $MX/out/mutate_mixo_train.out) RED; $(grep -aciE '^SURVIV|^GREEN' $MX/out/mutate_mixo_train.out) survivors; $(grep -a 'byte-identical' $MX/out/mutate_mixo_train.out | tail -1)"
echo ZZTV-DONE
