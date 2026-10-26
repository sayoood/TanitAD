#!/usr/bin/env bash
# MIXO proof re-check on the FINAL bytes: tip + OACC Amendment 2 (trainer anchor de-dup, reader, test, mutant) + the MIXO killing tests and anchors.
# The 162/162 proof ran on 6df75ad2, whose trainer predates O-RECOVER (98713259); every other MIXO-mutated file is byte-identical on the tip.
# So: --check (every anchor exactly once), the 15 trainer mutants, and the coupled label-pin check. Each pytest run waits for < 2 pytest RUNS.
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
TIP=$1
X=E:/Projects/TanitAD-artifacts/mx
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
G="git -C C:/Users/Admin/tanitad-push -c safe.directory=* -c core.autocrlf=false"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
OF=E:/Projects/TanitAD-artifacts/oaccfix/tree
MK=C:/Users/Admin/mixo_kill/tree
export PYTHONPATH="$X/tree/stack;$X/tree/taniteval;$X/tree/stack/scripts"
export MIXO_MUT_BASE=$X/t TMP=E:/Projects/TanitAD-artifacts/mx/st TEMP=E:/Projects/TanitAD-artifacts/mx/st
mkdir -p $X/out $X/st
slot() { for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && return 0; sleep 30; done; return 1; }
if [ ! -d $X/tree ]; then
  mkdir -p $X/tree
  $G archive $TIP -- stack taniteval "$PK" "Project Steering/PREREG_V8F_MIXO.md" "Project Steering/PREREG_V8F_OACC.md" "Project Steering/PREREG_V8F_FLAT.md" | tar -x -C $X/tree || { echo ZZMX-ARCHIVE-FAILED; exit 2; }
fi
for f in stack/scripts/v8f_mini_train.py stack/tests/test_v8f_oacc.py "$PK/code/v8f_oacc_read.py" "$PK/raw/2026-10-25-oacc/mutate_oacc.py"; do cp "$OF/$f" "$X/tree/$f"; done
for f in stack/tests/test_v8f_mixo.py "$PK/raw/2026-10-25-mixo/mutate_mixo.py"; do cp "$MK/$f" "$X/tree/$f"; done
cd $X/tree
for f in stack/scripts/v8f_mini_train.py stack/tests/test_v8f_oacc.py "$PK/code/v8f_oacc_read.py" "$PK/raw/2026-10-25-oacc/mutate_oacc.py" stack/tests/test_v8f_mixo.py "$PK/raw/2026-10-25-mixo/mutate_mixo.py"; do
  a=$($G rev-parse "$TIP:$f"); b=$(git hash-object --no-filters "$f"); echo "$f: $([ "$a" = "$b" ] && echo 'tip bytes' || echo 'changed vs tip')"
done > $X/out/tree_files.out
M="$PK/raw/2026-10-25-mixo/mutate_mixo.py"
"$PY" "$M" "$X/tree" "$PY" --check > $X/out/mutate_mixo_check.out 2>&1; echo "mixo check rc=$? $(tail -1 $X/out/mutate_mixo_check.out)"
slot || { echo ZZMX-NO-SLOT; exit 3; }
"$PY" -m pytest -q -p no:cacheprovider --basetemp $X/t/c stack/tests/test_v8f_flat.py stack/tests/test_v8f_oacc.py -k every_existing_label_is_byte_identical > $X/out/coupled_check.out 2>&1
echo "coupled rc=$? $(tail -1 $X/out/coupled_check.out)"
slot || { echo ZZMX-NO-SLOT; exit 3; }
"$PY" "$M" "$X/tree" "$PY" --only v8f_mini_train.py > $X/out/mutate_mixo_train.out 2>&1
echo "mixo trainer mutants rc=$? $(grep -ac '^RED' $X/out/mutate_mixo_train.out) RED; $(grep -aciE '^SURVIV|^GREEN' $X/out/mutate_mixo_train.out) survivors; $(grep -a 'byte-identical' $X/out/mutate_mixo_train.out | tail -1)"
echo ZZMX-DONE
