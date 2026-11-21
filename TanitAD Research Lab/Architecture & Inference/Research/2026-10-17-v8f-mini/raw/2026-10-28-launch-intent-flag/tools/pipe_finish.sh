#!/bin/bash
# waits for the mutation sweep and the suites, then assembles RESULTS.md / MANIFEST.txt / SHA256SUMS.txt / logs/ and writes FINISHED.txt. Single instance.
S=F:/TanitAD-scratch/launch_intent
O=E:/Projects/TanitAD-artifacts/dagland/launch_intent
mkdir $S/tmp/lock_finish 2>/dev/null || { echo "pipe_finish already running or done"; exit 0; }
until [ -f $S/logs/mut.done ] && [ -f $S/logs/suites.done ]; do sleep 60; done
C:/Users/Admin/venvs/tanitad/Scripts/python.exe $O/tools/assemble.py > $S/logs/assemble.out 2>&1
python $O/tools/enumerate_mutation.py $O/tools/enumerate_cells_li.py $O/enumerate_cells_li_mutation.txt > $S/logs/enum_mut.out 2>&1
C:/Users/Admin/venvs/tanitad/Scripts/python.exe $O/tools/assemble.py >> $S/logs/assemble.out 2>&1
date +%FT%T > $O/FINISHED.txt
