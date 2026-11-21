#!/bin/bash
export PATH="/usr/bin:/bin:$PATH"
L=F:/TanitAD-scratch/compose_tmp/logs
until [ -f $L/suites.done ] && [ -f $L/mut.done ]; do sleep 30; done
C:/Python314/python.exe F:/TanitAD-scratch/compose_tmp/assemble.py > $L/assemble.out 2>&1
echo finished > E:/Projects/TanitAD-artifacts/dagland/ego_dagger_compose/FINISHED.txt
