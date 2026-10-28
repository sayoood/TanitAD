#!/bin/bash
# DETACHED pipeline B: the mutation sweep (27 mutants, 11 controls) on the COPY F:/TanitAD-scratch/compose_fix/mut. Starts 60 s after pipeline A so the two do not race for the pytest slot.
export PATH="/usr/bin:/bin:$PATH"
L=F:/TanitAD-scratch/compose_fix/logs
D=E:/Projects/TanitAD-artifacts/dagland/compose_fix
mkdir -p $L
rm -f $L/mut.done
sleep 60
C:/Users/Admin/venvs/tanitad/Scripts/python.exe F:/TanitAD-scratch/compose_fix/tmp/run_mutations.py F:/TanitAD-scratch/compose_fix/mut $D/mutation.txt > $L/run_mutations.out 2>&1
echo done > $L/mut.done
