#!/bin/bash
# Control C8, IA-3 half (MEASURED when run): R35_s0 with the level-1 action embedding zeroed (weights and bias), through the SAME ia_read.py as the arms. Starts after the intact IA read of C7 exists (the intact control).
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
source /f/TanitAD-scratch/actsens/read/env.sh
O=$RD/out/C8; mkdir -p $O
while [ ! -f $RD/out/C7_R35_s0/ia_read.json ]; do sleep 60; done
OMP_NUM_THREADS=3 $PY -u $RD/ia_read.py --name C8_R35_s0 --run-dir $PULLS/ladder_R35_s0 --chassis-seed 0 --zero-act --out $O/ia_zeroact.json </dev/null > $O/ia_zeroact.log 2>&1
echo "rc=$? $(date -Iseconds)" > $O/C8_IA.done
$PY $RD/c8_check.py $RD/out/C7_R35_s0/ia_read.json $O/ia_zeroact.json $RD/out/C8/c8_result.json </dev/null >> $O/ia_zeroact.log 2>&1
