#!/bin/bash
# run_checks.sh -- the RHK build's CPU checks, ONE python at a time, each gated on commit >= 5.5 GB. Logs in this folder. Run from build/b9rhk.
B=F:/TanitAD-scratch/rollhist/build/b9rhk
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export TMP=$B/tmp TEMP=$B/tmp
cd $B || exit 1
gate() { $PY -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 || { echo "commit gate failed" >> run_checks.log; exit 1; }; }
echo "start $(date -u +%FT%TZ)" > run_checks.log
cd test
gate; $PY -I -m pytest -q -p no:cacheprovider --basetemp=$B/tmp/bt_full test_dscale10k_check_rhk.py test_rhk_argv.py test_rhk_pair_and_bank.py > ../pytest_log_k.txt 2>&1; echo "full rc=$?" >> ../run_checks.log
cd ..
gate; $PY -I make_mutants.py dscale10k_check_rhk.py $B/tmp/mut >> run_checks.log 2>&1
cd test
for m in mut_nodose mut_notoken mut_nopath; do
  gate; DS_CHECK_RH_PATH=$B/tmp/mut/$m.py $PY -I -m pytest -q -p no:cacheprovider --basetemp=$B/tmp/bt_$m test_dscale10k_check_rhk.py -k test_k_ > ../pytest_log_k_$m.txt 2>&1; echo "$m rc=$?" >> ../run_checks.log
done
gate; DS_CHECK_RH_PATH=$B/base_dscale10k_check_rhp3.py $PY -I -m pytest -q -p no:cacheprovider --basetemp=$B/tmp/bt_old test_dscale10k_check_rhk.py -k test_k_ > ../pytest_log_k_old_checker.txt 2>&1; echo "old checker (dcee7cc1) rc=$?" >> ../run_checks.log
echo "end $(date -u +%FT%TZ)" >> ../run_checks.log
