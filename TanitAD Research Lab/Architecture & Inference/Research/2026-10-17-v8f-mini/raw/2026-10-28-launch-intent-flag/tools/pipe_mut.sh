#!/bin/bash
# the mutation sweep of the trainer flag (30 mutants + one control per distinct selection). Single instance (mkdir lock). Writes logs/mut.done.
S=F:/TanitAD-scratch/launch_intent
mkdir $S/tmp/lock_mut 2>/dev/null || { echo "pipe_mut already running or done (remove $S/tmp/lock_mut to rerun)"; exit 0; }
C:/Users/Admin/venvs/tanitad/Scripts/python.exe E:/Projects/TanitAD-artifacts/dagland/launch_intent/tools/run_mutations.py $S/mut E:/Projects/TanitAD-artifacts/dagland/launch_intent/mutation.txt
echo "rc=$?" > $S/logs/mut.done
