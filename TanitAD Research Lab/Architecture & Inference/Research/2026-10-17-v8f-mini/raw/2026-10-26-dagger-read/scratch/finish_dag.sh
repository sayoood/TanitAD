#!/bin/bash
# Waits (as a script, no model turns) for the mutation run to end, then closes out the evidence. Run from C:/Users/Admin/dagread.
cd C:/Users/Admin/dagread || exit 1
until grep -q "^mutation rc=" out/mutation_console.out 2>/dev/null; do sleep 30; done
sha256sum -c out/snap_before_SHA256.txt > out/snap_after_check.out 2>&1; SNAP_RC=$?
LAST=$(tail -1 out/mutation.out)
PYT=$(tail -1 out/pytest_full.out)
{
  echo
  echo "## Evidence (written by scratch/finish_dag.sh after the mutation run)"
  echo "* \`out/pytest_full.out\`: ${PYT}"
  echo "* \`out/mutation.out\` (\`out/mutation_console.out\` for the live log): ${LAST}"
  if [ "$SNAP_RC" = "0" ]; then echo "* R77 snapshot: dag_read.py, test_dag_read.py, mutate_dag.py and the PREREG copy are BYTE-IDENTICAL after the mutation run (\`out/snap_before_SHA256.txt\`, \`out/snap_after_check.out\`)."; else echo "* R77 snapshot: MISMATCH after the mutation run, see \`out/snap_after_check.out\` -- the deliverables changed; do not land."; fi
} >> NOTES.md
sha256sum dag_read.py test_dag_read.py mutate_dag.py NOTES.md PREREG_V8F_DAGGER.md > out/final_SHA256.txt
echo "finish_dag done snap_rc=$SNAP_RC" > out/finish_dag.done
