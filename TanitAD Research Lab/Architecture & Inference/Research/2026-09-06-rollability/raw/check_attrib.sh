set -u
WT=/c/Users/Admin/tanitad-wt
T="$WT/stack/tanitad/refs/refc_v3.py"
SEL="stack/tests/test_runbook_commands.py stack/tests/test_kingate_contract.py stack/tests/test_build_parity_guard.py"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8
export PYTHONPATH="C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/taniteval"
cp "$T" /c/Users/Admin/_roll/_fixed_backup.py
echo "--- FIXED ---"
( cd "$WT" && "$PY" -m pytest $SEL -q --no-header 2>&1 | tail -4 )
echo "--- DEFECT (pre-fix refc_v3.py restored) ---"
cp /c/Users/Admin/_roll/refc_v3.py.orig "$T"
( cd "$WT" && "$PY" -m pytest $SEL -q --no-header 2>&1 | tail -4 )
cp /c/Users/Admin/_roll/_fixed_backup.py "$T"
a=$(md5sum "$T" | cut -d' ' -f1); b=$(md5sum /c/Users/Admin/_roll/refc_v3.py.new | cut -d' ' -f1)
if [ ${#a} -ne 32 ] || [ ${#b} -ne 32 ]; then echo "RESTORE INCONCLUSIVE"; elif [ "$a" = "$b" ]; then echo "RESTORE VERIFIED $a"; else echo "RESTORE MISMATCH"; fi
