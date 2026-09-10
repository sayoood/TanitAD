#!/usr/bin/env bash
# Wait for b1's ARTIFACT (the DONE marker + idx.json), never for an exit code
# (CLAUDE.md: assert on the artifact, not the status). Then run the A3 panel.
LOG=/c/Users/Admin/wpd-probe/bank_rep.log
IDX=/c/Users/Admin/wpd-probe/bank_rep/idx.json
for i in $(seq 1 400); do
  if [ -f "$IDX" ] && grep -q "^DONE " "$LOG" 2>/dev/null; then
    echo "BANK_ARTIFACT_PRESENT after ${i} polls"; break
  fi
  if grep -qE "Traceback|MemoryError|CUDA out of memory|AssertionError" "$LOG" 2>/dev/null; then
    echo "BANK_FAILED"; tail -30 "$LOG"; exit 3
  fi
  sleep 5
done
[ -f "$IDX" ] || { echo "BANK_NEVER_FINISHED"; tail -20 "$LOG"; exit 4; }
tail -22 "$LOG"
exec /c/Users/Admin/wpd-probe/code/b10_run_a3.sh
