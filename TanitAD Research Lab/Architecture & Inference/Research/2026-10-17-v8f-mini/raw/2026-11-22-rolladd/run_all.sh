#!/bin/bash
# run_all.sh -- ROLLADD: every CPU job, one python process at a time, each gated on commit >= 5.5 GB (wait_commit). Logs under F:/TanitAD-scratch/rolladd/. No GPU. Nothing written to the pod.
R=F:/TanitAD-scratch/rolladd
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
G="$PY -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 240"
export RA_CENSUS_WM_OUT=$R/WM_SIGREG_CENSUS.json
bash $R/run_tests.sh $R/test_log_rolladd_new.txt stack/tests/test_v8f_rolladd.py
bash $R/run_tests.sh $R/test_log_rolladd_whole.txt stack/tests/test_v8f_dagger.py stack/tests/test_v8f_dagger_train.py stack/tests/test_v8f_ego_dagger_compose.py stack/tests/test_v8f_recovery.py stack/tests/test_v8f_recovery_labeller.py stack/tests/test_v8f_recovery_wiring.py stack/tests/test_v8f_rollhist.py stack/tests/test_v8f_rolladd.py
# pod-side tests (checker kind RA, gate record RA)
$G
cd $R/pod
mkdir -p $R/tmp/ptpod
env TMP=$R/tmp/ptpod TEMP=$R/tmp/ptpod CUDA_VISIBLE_DEVICES=-1 $PY -I -m pytest -q -p no:cacheprovider --basetemp=$R/tmp/ptpod/bt test/test_ra_checker.py > $R/test_log_ra_checker.txt 2>&1; echo "RC $?" >> $R/test_log_ra_checker.txt
$G
env TMP=$R/tmp/ptpod TEMP=$R/tmp/ptpod $PY -I test_ra_gate_record.py $R/pod/ra_gate_record.py $R/tmp/ptpod/grt > $R/test_log_ra_gate_record.txt 2>&1; echo "RC $?" >> $R/test_log_ra_gate_record.txt
# checker mutants: each must turn test_ra_checker.py RED
for m in $R/mutants/checker/mut_*.py; do
  $G
  n=$(basename $m .py)
  env TMP=$R/tmp/ptpod TEMP=$R/tmp/ptpod DS_CHECK_RA_PATH=$(cygpath -w $m) $PY -I -m pytest -q -x -p no:cacheprovider --basetemp=$R/tmp/ptpod/bt_$n test/test_ra_checker.py > $R/mutants/log_$n.txt 2>&1; echo "$n rc=$?" >> $R/mutants/checker_result.txt
done
# real-bank census
$G
env TMP=$R/tmp/tt TEMP=$R/tmp/tt CUDA_VISIBLE_DEVICES=-1 $PY -I $R/census_rolladd.py > $R/real_bank_census.log 2>&1
# code mutants
$PY -I $R/mutants/run_mutants.py > $R/mutants/run1.log 2>&1
$PY -I $R/mutants/run_mutants.py restore >> $R/mutants/run1.log 2>&1
echo ALLDONE > $R/run_all.done
