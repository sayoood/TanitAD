#!/bin/bash
# Detached closer for the A1 pass (no model turns): R77 snapshot -> full mutation proof on COPIES -> byte-identity re-check -> evidence. Run from C:/Users/Admin/dagread.
cd C:/Users/Admin/dagread || exit 1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
FILES="dag_read.py test_dag_read.py mutate_dag.py bank_audit.py PREREG_V8F_DAGGER.md fixtures/dagger_smoke/PROVENANCE.json fixtures/dagger_smoke/bank.json fixtures/dagger_smoke/shard_000of001.provenance.json"
sha256sum $FILES > out/snap_before_SHA256.txt
date > out/mutation_started.txt
$PY mutate_dag.py --out out/mutation.out > out/mutation_console.out 2>&1
echo "mutation rc=$?" >> out/mutation_console.out
sha256sum -c out/snap_before_SHA256.txt > out/snap_after_check.out 2>&1; SNAP_RC=$?
LAST=$(tail -1 out/mutation.out)
PYT=$(tail -1 out/pytest_full.out)
{
  echo
  echo "## 8. Evidence of this pass (written by scratch/finish_a1.sh after the mutation run)"
  echo "* \`out/pytest_full.out\`: ${PYT}"
  echo "* \`out/mutation.out\` (live log \`out/mutation_console.out\`): ${LAST}"
  if [ "$SNAP_RC" = "0" ]; then echo "* R77: dag_read.py, test_dag_read.py, mutate_dag.py, bank_audit.py, the landed PREREG copy and the three fixtures are BYTE-IDENTICAL after the mutation run (\`out/snap_before_SHA256.txt\` vs \`out/snap_after_check.out\`; mutants ran on copies under E:/Projects/TanitAD-artifacts/dagread_build/mut)."; else echo "* R77: MISMATCH after the mutation run, see \`out/snap_after_check.out\` -- do not land."; fi
  echo "* sha256 before the mutation run: see \`out/snap_before_SHA256.txt\`; after: \`out/final_SHA256.txt\`."
} >> NOTES.md
sha256sum $FILES NOTES.md > out/final_SHA256.txt
echo "finish_a1 done snap_rc=$SNAP_RC" > out/finish_a1.done
