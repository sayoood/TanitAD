#!/bin/bash
# PINS_LEVEL.txt: one PIN line per file pinned in in_pod/arms_actcons_c0.txt (hash taken in work/tree_level), plus the new test file; copies of the changed files into patched/; ACTLEVEL.patch.
cd /f/TanitAD-scratch/v8proof/actlevel || exit 1
T=/workspace/v8proof/tree_actlevel
{
  grep '^PIN|' in_pod/arms_actcons_c0.txt | cut -d'|' -f3 | while read -r f; do
    echo "PIN|$T|$f|$(sha256sum work/tree_level/$f | cut -d' ' -f1)"
  done
  echo "PIN|$T|stack/tests/test_v8f_act_level.py|$(sha256sum work/tree_level/stack/tests/test_v8f_act_level.py | cut -d' ' -f1)"
} > PINS_LEVEL.txt
mkdir -p patched/scripts patched/tanitad/models patched/tests
cp work/tree_level/stack/scripts/v8f_mini_train.py patched/scripts/
cp work/tree_level/stack/tanitad/models/act_cons.py patched/tanitad/models/
cp work/tree_level/stack/tests/test_v8f_act_cons.py work/tree_level/stack/tests/test_v8f_act_level.py patched/tests/
OLD=/f/TanitAD-scratch/v8proof/actcons/work/tree_patch
: > ACTLEVEL.patch
for f in stack/tanitad/models/act_cons.py stack/scripts/v8f_mini_train.py stack/tests/test_v8f_act_cons.py; do
  diff -u --label "a/$f" --label "b/$f" "$OLD/$f" "work/tree_level/$f" >> ACTLEVEL.patch
done
diff -u --label /dev/null --label "b/stack/tests/test_v8f_act_level.py" /dev/null work/tree_level/stack/tests/test_v8f_act_level.py >> ACTLEVEL.patch
wc -l PINS_LEVEL.txt ACTLEVEL.patch
