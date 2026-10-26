#!/usr/bin/env bash
# After the anchor de-duplication in v8f_mini_train.py (two lines respelled, the ego probe block moved after the model returns to its device):
# both affected test files, the OACC acc-probe / lever / weight mutants, and the FULL O-RECOVER mutation proof. Each waits for < 2 pytest RUNS.
export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
X=E:/Projects/TanitAD-artifacts/oaccfix
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
export PYTHONPATH="$X/tree/stack;$X/tree/taniteval;$X/tree/stack/scripts"
slot() { for i in $(seq 1 720); do n=$("$PY" C:/Users/Admin/orcread/pytest_runs.py | tr -dc '0-9'); [ -n "$n" ] && [ "$n" -lt 2 ] && return 0; sleep 30; done; return 1; }
slot || { echo ZZT2-NO-SLOT; exit 3; }
cd $X/tree/stack
echo "tests start $(date '+%H:%M:%S')"
"$PY" -m pytest -q -p no:cacheprovider --basetemp $X/pt/s2 tests/test_v8f_oacc.py tests/test_v8f_ego_perturb.py > $X/out/pytest_oacc_ego.out 2>&1
echo "tests rc=$? $(tail -1 $X/out/pytest_oacc_ego.out)"
cd $X/tree
M="$PK/raw/2026-10-25-oacc/mutate_oacc.py"
"$PY" "$M" "$X/tree" "$PY" --check > $X/out/mutate_oacc_check.out 2>&1; echo "oacc check rc=$? $(tail -1 $X/out/mutate_oacc_check.out)"
for sel in "acc probe" "built levers" "loss weights"; do
  slot || { echo ZZT2-NO-SLOT; exit 3; }
  tag=$(echo "$sel" | tr ' ' '_')
  "$PY" "$M" "$X/tree" "$PY" --only "$sel" --tmp $X/pt/m_$tag > $X/out/mutate_oacc_$tag.out 2>&1
  echo "oacc mutants '$sel' rc=$? $(grep -c 'RED\|KILLED' $X/out/mutate_oacc_$tag.out) red-lines; $(grep -ci 'SURVIV\|GREEN ' $X/out/mutate_oacc_$tag.out) survivor-lines"
done
slot || { echo ZZT2-NO-SLOT; exit 3; }
"$PY" "$PK/raw/2026-10-26-orecover/mutate_orecover.py" "$X/tree" "$PY" > $X/out/mutate_orecover.out 2>&1
echo "orecover proof rc=$? $(tail -2 $X/out/mutate_orecover.out | tr '\n' ' ')"
cd $X/tree && for f in stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_mini_data.py; do git -C /c/Users/Admin/tanitad-push -c safe.directory='*' show 98713259:$f | diff - $f > $X/out/$(basename $f).vs98713259.diff; echo "$f: $(grep -c '^[<>]' $X/out/$(basename $f).vs98713259.diff) changed lines vs 98713259"; done
echo ZZT2-DONE
