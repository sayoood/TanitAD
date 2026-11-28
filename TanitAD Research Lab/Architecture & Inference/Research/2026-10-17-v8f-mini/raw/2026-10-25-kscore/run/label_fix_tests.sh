#!/bin/bash
# The KSCORE strategic-label fix (R73): the full test file on the fixed tree, then the mutation (the tip's old label restored) must go RED.
SC=C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad
R=$SC/kfix; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; O=$SC/kscore_res
cd $R/stack || exit 1
rm -rf $SC/kfix_bt; echo "== fixed tree (tip 4f561a2c + the label fix)"
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$R/stack timeout 900 $PY -m pytest -q -p no:cacheprovider --basetemp=$SC/kfix_bt tests/test_v8f_kscore.py 2>&1 | tail -3
cp scripts/v8f_kscore.py $O/v8f_kscore.fixed.py
git -C C:/Users/Admin/tanitad-push -c safe.directory=* show 4f561a2c1942a41be1150aca9e5a6e3f0748fe79:stack/scripts/v8f_kscore.py > scripts/v8f_kscore.py
echo "== MUTATION: the tip's old label restored (expect RED on the result-text test)"
rm -rf $SC/kfix_bt
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$R/stack timeout 900 $PY -m pytest -q -p no:cacheprovider --basetemp=$SC/kfix_bt tests/test_v8f_kscore.py -k "$1" 2>&1 | grep -E "passed|failed|AssertionError|^E " | head -6
cp $O/v8f_kscore.fixed.py scripts/v8f_kscore.py && cmp -s $O/v8f_kscore.fixed.py scripts/v8f_kscore.py && echo "== fixed file restored (cmp OK)"
rm -rf $SC/kfix_bt
