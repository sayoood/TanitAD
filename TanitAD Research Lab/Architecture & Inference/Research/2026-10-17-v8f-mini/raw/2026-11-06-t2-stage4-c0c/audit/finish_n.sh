#!/usr/bin/env bash
# AGENT N / O close-out chain (detached): the full suite (log name = the one the read's G2(b) cites), then PINS_N and the verdict file.
# The mutation run is a separate detached job (logs/mutations_stage4.log); this chain does NOT rebuild the tar (no packaged file changed: BUILD_RESULT4.json says which).
cd E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f || exit 9
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1
mkdir -p /c/Users/Admin/pytest_tmp_O
# The rehearsal fixtures (~1 GB each) and the drivers' 10 GiB free-space floor: C: had 9.9 GiB free at 19:25, so the scratch goes on F: (NTFS, 19 GB free, hard links work). Nothing is deleted.
export T2V8F_S1_SCRATCH=F:/TanitAD-scratch/t2v8f_rehearsal_O/stage1
export T2V8F_S2_SCRATCH=F:/TanitAD-scratch/t2v8f_rehearsal_O/stage2
mkdir -p "$T2V8F_S1_SCRATCH" "$T2V8F_S2_SCRATCH"
$PY -I -m pytest tests -q -p no:cacheprovider --basetemp=/c/Users/Admin/pytest_tmp_O/full > logs/pytest_full_M.log 2>&1
echo "full suite rc=$?" > logs/N_FINISHED.txt
tail -3 logs/pytest_full_M.log >> logs/N_FINISHED.txt
tail -2 logs/mutations_stage4.log >> logs/N_FINISHED.txt
$PY -I audit/make_pins_n.py > /dev/null 2>&1
echo "pins rc=$?" >> logs/N_FINISHED.txt
echo "ZZ N-CHAIN-DONE ZZ" >> logs/N_FINISHED.txt
