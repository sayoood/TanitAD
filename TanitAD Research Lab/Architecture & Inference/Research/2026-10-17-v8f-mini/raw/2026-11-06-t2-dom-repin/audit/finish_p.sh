#!/usr/bin/env bash
# AGENT P close-out chain (detached): the full suite (log logs/pytest_full_P.log), the R-SCN mutation run (logs/mutations_dom_p.log), then PINS_P and the verdict file.
# No packaged file changed (the read is not in the stage-4 overlay), so nothing is rebuilt. Scratch goes on F: (C: is low); nothing is deleted.
cd E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f || exit 9
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1
export T2V8F_S1_SCRATCH=F:/TanitAD-scratch/t2v8f_rehearsal_P/stage1
export T2V8F_S2_SCRATCH=F:/TanitAD-scratch/t2v8f_rehearsal_P/stage2
export P_MUT_SCRATCH=F:/TanitAD-scratch/t2v8f_P/mut_final
mkdir -p "$T2V8F_S1_SCRATCH" "$T2V8F_S2_SCRATCH" F:/TanitAD-scratch/t2v8f_P
echo "started $(date +%H:%M:%S)" > logs/P_FINISHED.txt
$PY -I -m pytest tests -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/t2v8f_P/full > logs/pytest_full_P.log 2>&1
echo "full suite rc=$?" >> logs/P_FINISHED.txt
tail -3 logs/pytest_full_P.log >> logs/P_FINISHED.txt
$PY -I code/run_mutations_dom_p.py > logs/mutations_dom_p.log 2>&1
echo "mutations rc=$?" >> logs/P_FINISHED.txt
tail -3 logs/mutations_dom_p.log >> logs/P_FINISHED.txt
$PY -I audit/make_pins_p.py > /dev/null 2>&1
echo "pins rc=$?" >> logs/P_FINISHED.txt
echo "ZZ P-CHAIN-DONE ZZ" >> logs/P_FINISHED.txt
