#!/usr/bin/env bash
# MM: whole-file, dagger-first re-run of the RECLAB test files at the landed commit cf27dbb7
set -u
R=/f/TanitAD-scratch/reclab
T=$R/tree_cf27
BT=$R/tmp/pt_cf27
LOG=$R/test_log_cf27dbb7_cpu.txt
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60
cd "$T"
mkdir -p /f/TanitAD-scratch/reclab/tmp/tt_cf27
export PYTHONPATH="$(cygpath -w "$T/stack")" CUDA_VISIBLE_DEVICES=-1 TMP=F:/TanitAD-scratch/reclab/tmp/tt_cf27 TEMP=F:/TanitAD-scratch/reclab/tmp/tt_cf27
env -u MSYS_NO_PATHCONV "$PY" -c "import tanitad,torch;print('tanitad from',tanitad.__file__,'cuda',torch.cuda.is_available())" > "$LOG" 2>&1
grep -q "cuda False" "$LOG" || { echo "REFUSED: CUDA is visible" >> "$LOG"; echo DONE >> "$LOG"; exit 1; }
env -u MSYS_NO_PATHCONV "$PY" -m pytest -q -p no:cacheprovider --basetemp="$(cygpath -w "$BT")" \
  stack/tests/test_v8f_dagger.py stack/tests/test_v8f_dagger_train.py stack/tests/test_v8f_ego_dagger_compose.py \
  stack/tests/test_v8f_recovery.py stack/tests/test_v8f_recovery_labeller.py stack/tests/test_v8f_recovery_wiring.py >> "$LOG" 2>&1
echo "RC $?" >> "$LOG"
du -sh "$BT" >> "$LOG" 2>&1
du -sh /f/TanitAD-scratch/reclab/tmp/tt_cf27 >> "$LOG" 2>&1
rm -rf "/f/TanitAD-scratch/reclab/tmp/pt_cf27" "/f/TanitAD-scratch/reclab/tmp/tt_cf27"
echo DONE >> "$LOG"
