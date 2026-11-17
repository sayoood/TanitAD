#!/bin/bash
cd F:/TanitAD-scratch/actrsp/optb/pkg
while [ ! -e F:/TanitAD-scratch/actrsp/optb/regress2.done ]; do sleep 30; done
C:/Users/Admin/venvs/tanitad/Scripts/python.exe F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 6 120
OPTB_TEST_TMP=C:/Users/Admin/AppData/Local/Temp/optb_dry_run_a bash tests/test_optb_dryrun.sh > F:/TanitAD-scratch/actrsp/optb/dryrun1.log 2>&1
echo "DRY-FINISHED rc=$?" >> F:/TanitAD-scratch/actrsp/optb/dryrun1.log
