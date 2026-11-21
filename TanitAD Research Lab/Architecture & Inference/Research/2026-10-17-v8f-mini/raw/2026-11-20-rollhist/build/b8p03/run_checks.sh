#!/bin/bash
# run_checks.sh -- the dev-box checks of the P03 build, ONE python process at a time, each gated on commit >= 5.5 GB and F: >= 6 GB. Logs under build/b8p03/. Writes only under build/b8p03 and read/out/read_p03|steps_p03 (never).
export PATH="/usr/bin:/bin:$PATH"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
B=F:/TanitAD-scratch/rollhist/build/b8p03
RD=F:/TanitAD-scratch/rollhist/read
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
export TMP=$B/tmp TEMP=$B/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8
L=$B/run_checks.log
gate() { $PY -I $WC 5.5 60 >> $L 2>&1 || { echo "GATE FAIL before $1" >> $L; exit 1; }
         fk=$(df -k /f | awk 'NR==2{print int($4/1048576)}'); [ "${fk:-0}" -ge 6 ] || { echo "F: ${fk} GB < 6 before $1" >> $L; exit 1; }; }
echo "start $(date -u +%T)" > $L
cd $B/test
gate pytest;  $PY -I -m pytest -q -p no:cacheprovider --basetemp=$B/tmp/bt1 . > $B/pytest_log_p3.txt 2>&1; echo "pytest rc $?" >> $L
# mutation arm 1: the dose pin removed from the new checker (any p > 0 admitted): the RHp3 tests must fail
sed 's/elif a.get("--dagger-p") != \[RHP3_P\]:/elif False:/' $B/dscale10k_check_rhp3.py > $B/tmp/mutant_nopin.py
cmp -s $B/dscale10k_check_rhp3.py $B/tmp/mutant_nopin.py && echo "MUTANT NOT APPLIED" >> $L
gate mutant1; DS_CHECK_RH_PATH=$B/tmp/mutant_nopin.py $PY -I -m pytest -q -p no:cacheprovider --basetemp=$B/tmp/bt2 test_dscale10k_check_rhp3.py > $B/pytest_log_p3_mutant.txt 2>&1; echo "mutant(no dose pin) rc $?" >> $L
# mutation arm 2: the OLD checker (69ff1a0f, no RHp3 kind) under the same tests: must fail
gate mutant2; DS_CHECK_RH_PATH=$B/pod_ro/dscale10k_check_rh.py $PY -I -m pytest -q -p no:cacheprovider --basetemp=$B/tmp/bt3 test_dscale10k_check_rhp3.py > $B/pytest_log_p3_old_checker.txt 2>&1; echo "old checker rc $?" >> $L
# the read: selftest under the default (the PARTIAL read's own), then under RH_AKIND=RHp3
cd $RD
gate selftest_default; env -u RH_AKIND $PY -u rh_read.py --selftest > $B/read_selftest_default.log 2>&1; echo "read selftest default rc $?" >> $L
gate selftest_p03; RH_AKIND=RHp3 $PY -u rh_read.py --selftest > $B/read_selftest_p03.log 2>&1; echo "read selftest RHp3 rc $?" >> $L
gate dry_check; RH_AKIND=RHp3 $PY -u dry_check.py > $B/dry_check_p03.log 2>&1; echo "dry_check RHp3 rc $?" >> $L
echo "ZZ RUN-CHECKS DONE ZZ $(date -u +%T)" >> $L
