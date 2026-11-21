#!/usr/bin/env bash
# AGENT R: the stage-5 rehearsal (detached). Scratch on E: (C: is near full). Ends with ZZ R-CHAIN-DONE ZZ in logs/R_FINISHED.txt.
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH"
cd E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f || exit 9
export CUDA_VISIBLE_DEVICES=-1
export T2V8F_S1_SCRATCH=E:/Projects/TanitAD-artifacts/scratch_s5_t2v8f/rehearsal/stage1
export T2V8F_S2_SCRATCH=E:/Projects/TanitAD-artifacts/scratch_s5_t2v8f/rehearsal/stage2
echo "started $(date +%H:%M:%S)" > logs/R_FINISHED.txt
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest tests/test_t2v8f_stage5.py -q -p no:cacheprovider --basetemp=E:/Projects/TanitAD-artifacts/scratch_s5_t2v8f/pt_main > logs/pytest_stage5_R.log 2>&1
echo "pytest rc=$?" >> logs/R_FINISHED.txt
tail -4 logs/pytest_stage5_R.log >> logs/R_FINISHED.txt
echo "ZZ R-CHAIN-DONE ZZ" >> logs/R_FINISHED.txt
