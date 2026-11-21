#!/usr/bin/env bash
# AGENT R: re-run of test_identity alone after the harness fix (fixture5 was not built before the stage-5 cell was called). Scratch on E:.
export PATH="/usr/bin:/bin:/mingw64/bin:$PATH"
cd E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f || exit 9
export CUDA_VISIBLE_DEVICES=-1
export T2V8F_S1_SCRATCH=E:/Projects/TanitAD-artifacts/scratch_s5_t2v8f/rehearsal/stage1
export T2V8F_S2_SCRATCH=E:/Projects/TanitAD-artifacts/scratch_s5_t2v8f/rehearsal/stage2
echo "started $(date +%H:%M:%S)" > logs/R2_FINISHED.txt
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest tests/test_t2v8f_stage5.py -q -p no:cacheprovider --tb=short -k "identity" --basetemp=E:/Projects/TanitAD-artifacts/scratch_s5_t2v8f/pt_id2 > logs/pytest_stage5_R2_identity.log 2>&1
echo "pytest rc=$?" >> logs/R2_FINISHED.txt
tail -4 logs/pytest_stage5_R2_identity.log >> logs/R2_FINISHED.txt
echo "ZZ R2-CHAIN-DONE ZZ" >> logs/R2_FINISHED.txt
