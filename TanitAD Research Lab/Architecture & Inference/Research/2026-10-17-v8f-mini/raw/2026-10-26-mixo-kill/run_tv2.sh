#!/usr/bin/env bash
# PREREG_V8F_ORECOVER Amendment 4 (853f2fc3) changed the trainer after run_tv.sh re-checked MIXO on 7f06ff00's (15/15 RED, out/on_7f06ff00/).
# So the MIXO trainer check runs again on the tip's trainer: --check, the coupled check, the 15 trainer mutants. Starts after run_lora.sh
# (the slot poll alone races); every pytest run waits for < 2 pytest RUNS.
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
TIP=$1
MX=E:/Projects/TanitAD-artifacts/mx
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
G="git -C C:/Users/Admin/tanitad-push -c safe.directory=* -c core.autocrlf=false"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
slot() { for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && return 0; sleep 30; done; return 1; }
O=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/tasks/bn8hv6jt3.output
for i in $(seq 1 720); do grep -q "^lora_ar2 rc=\|ZZLORA-NO-SLOT" $O 2>/dev/null && break; sleep 30; done
[ "$(git hash-object --no-filters $MX/tree/stack/scripts/v8f_mini_train.py)" = "$($G rev-parse $TIP:stack/scripts/v8f_mini_train.py)" ] || { echo ZZTV2-MX-TRAINER-NOT-TIP; exit 4; }
export PYTHONPATH="$MX/tree/stack;$MX/tree/taniteval;$MX/tree/stack/scripts" MIXO_MUT_BASE=$MX/t3 TMP=$MX/st TEMP=$MX/st
mkdir -p $MX/t3 $MX/st $MX/t
cd $MX/tree
for f in stack/scripts/v8f_mini_train.py stack/tests/test_v8f_mixo.py stack/tests/test_v8f_ego_perturb.py "$PK/raw/2026-10-25-mixo/mutate_mixo.py"; do echo "$f: $([ "$($G rev-parse "$TIP:$f")" = "$(git hash-object --no-filters "$f")" ] && echo 'tip bytes' || echo 'changed vs tip')"; done > $MX/out/tree_files.out
M="$PK/raw/2026-10-25-mixo/mutate_mixo.py"
"$PY" "$M" "$MX/tree" "$PY" --check > $MX/out/mutate_mixo_check.out 2>&1; echo "mixo check rc=$? $(tail -1 $MX/out/mutate_mixo_check.out)"
slot || { echo ZZTV2-NO-SLOT; exit 3; }
"$PY" -m pytest -q -p no:cacheprovider --basetemp $MX/t/c4 stack/tests/test_v8f_flat.py stack/tests/test_v8f_oacc.py -k every_existing_label_is_byte_identical > $MX/out/coupled_check.out 2>&1
echo "mixo coupled rc=$? $(tail -1 $MX/out/coupled_check.out)"
slot || { echo ZZTV2-NO-SLOT; exit 3; }
"$PY" "$M" "$MX/tree" "$PY" --only v8f_mini_train.py > $MX/out/mutate_mixo_train.out 2>&1
echo "mixo trainer mutants rc=$? $(grep -ac '^RED' $MX/out/mutate_mixo_train.out) RED; $(grep -aciE '^SURVIV|^GREEN' $MX/out/mutate_mixo_train.out) survivors; $(grep -a 'byte-identical' $MX/out/mutate_mixo_train.out | tail -1)"
echo ZZTV2-DONE
