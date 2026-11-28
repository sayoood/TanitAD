#!/bin/bash
# run_all2.sh -- after run_all.sh: the read extension's dry checks (default and RA) and the MECH2 control (mech_ra.py on RL's T0 evals planted under the RA names must reproduce MECH_RECLAB.json). CPU, gated, one process at a time.
R=F:/TanitAD-scratch/rolladd
E=F:/TanitAD-scratch/rollhist/read
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
G="$PY -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 240"
until [ -f $R/run_all.done ]; do sleep 20; done
cd $E
$G
env -u RH_AKIND $PY -I dry_check.py > $R/dry_check_default.log 2>&1; echo "RC $?" >> $R/dry_check_default.log
$G
env RH_AKIND=RA $PY -I dry_check.py > $R/dry_check_RA.log 2>&1; echo "RC $?" >> $R/dry_check_RA.log
# MECH2 control: RL's T0 evals planted under the RA names (own temp, deleted after)
for s in 0 1; do cp $E/out/t0/eval_RL_s$s.npz $E/out/t0/eval_RA_s$s.npz; done
$G
$PY -I $E/mech_ra.py $E $R/MECH2_CONTROL_RL_AS_RA.json > $R/mech2_control.log 2>&1; echo "RC $?" >> $R/mech2_control.log
for s in 0 1; do rm -f "F:/TanitAD-scratch/rollhist/read/out/t0/eval_RA_s$s.npz"; echo "F:/TanitAD-scratch/rollhist/read/out/t0/eval_RA_s$s.npz (planted copy of eval_RL_s$s.npz, MECH2 control)" >> $R/DELETIONS.log; done
echo ALLDONE2 > $R/run_all2.done
