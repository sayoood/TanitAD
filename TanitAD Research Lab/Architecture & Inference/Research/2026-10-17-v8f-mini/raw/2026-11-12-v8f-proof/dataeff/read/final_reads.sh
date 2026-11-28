#!/usr/bin/env bash
# final_reads.sh : the four-family reads at T1-kin with six baseline seeds (SPATIAL Q3 of card ADD12, DATAEFF of card ADD13). CPU, zero training.
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof
ROOTS=$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
cd /f/TanitAD-scratch/v8proof/spatial/q3 || exit 1
run() { # name out-dir pairs seeds extra
  echo "==== $1 start $(date -u +%H:%M:%SZ)"
  "$PY" q3_read.py read --t1-root "$ROOTS" --pairs "$3" --seeds "$4" --spread-arm HIER --spread-seeds 0,1,2,3,4,5 $5 --out-dir "$2" > "$2.log" 2>&1
  echo "==== $1 rc $? $(date -u +%H:%M:%SZ): $(tail -n 2 "$2.log" | tr '\n' ' ' | cut -c1-300)"
}
mkdir -p $W/spatial/q3/out_final $W/dataeff/read/out_final $W/dataeff/read/out_final_eighth_choice
run SPATIAL-Q3 $W/spatial/q3/out_final "G:HIER,W:HIER,G:W,P:G,P:HIER" 0,1 ""
run DATAEFF $W/dataeff/read/out_final "R1_2:HIER,R1_4:HIER,R1_8:HIER,R1_8:R1_4,S1_4:R1_4,R1_4:F1_4,HIER:FLAT" 0,1 "--same-sign"
run DATAEFF-EIGHTH-CHOICE-s0 $W/dataeff/read/out_final_eighth_choice "S1_8:R1_8,R1_8:F1_8" 0 "--same-sign"
echo "ZZ FINAL READS DONE $(date -u +%H:%M:%SZ)"
