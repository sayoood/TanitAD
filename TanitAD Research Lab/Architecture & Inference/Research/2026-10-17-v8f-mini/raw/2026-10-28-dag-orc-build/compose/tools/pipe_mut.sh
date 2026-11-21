#!/bin/bash
export PATH="/usr/bin:/bin:$PATH"
L=F:/TanitAD-scratch/compose_tmp/logs
D=E:/Projects/TanitAD-artifacts/dagland/ego_dagger_compose
rm -f $L/mut.done
C:/Python314/python.exe F:/TanitAD-scratch/compose_tmp/run_mutations.py F:/TanitAD-scratch/compose_mut $D/mutation.txt > $L/run_mutations.out 2>&1
echo done > $L/mut.done
