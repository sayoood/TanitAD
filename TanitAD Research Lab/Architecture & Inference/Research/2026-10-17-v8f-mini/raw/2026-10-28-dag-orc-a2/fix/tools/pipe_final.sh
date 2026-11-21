#!/bin/bash
# DETACHED pipeline C: when A and B are done, assemble the deliverable (stage/, diff.patch, RESULTS.md, SHA256SUMS) and write FINISHED.txt.
export PATH="/usr/bin:/bin:$PATH"
L=F:/TanitAD-scratch/compose_fix/logs
D=E:/Projects/TanitAD-artifacts/dagland/compose_fix
until [ -f $L/suites.done ] && [ -f $L/mut.done ]; do sleep 30; done
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/compose_fix/tmp/assemble.py > $L/assemble.out 2>&1
RC=$?
echo "finished $(date +%Y-%m-%dT%H:%M:%S)  (assemble rc=$RC)" > $D/FINISHED.txt
