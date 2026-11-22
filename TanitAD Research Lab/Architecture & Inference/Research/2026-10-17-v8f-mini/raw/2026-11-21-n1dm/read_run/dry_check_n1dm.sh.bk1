#!/bin/bash
# dry_check_n1dm.sh -- the N1DM read package's dry check, CPU, ONE python at a time, each gated on commit >= 5.5 GB. No dumps, no checkpoints needed.
#  1 py_compile of every read tool   2 bash -n of every shell script   3 test_n1dm_outcome.py (pure outcome / guard logic)   4 test_n1dm_paths.py (pod dump names == the read's dump paths; Z-USE / Z-UNIF arms; no stale tag)
#  5 eval141/dry_check.py (the eval-141 chain's own dry check; needs the N1DM tree at N1DM_TREE and the banked N1D inputs)
# Output: dry_check_n1dm.log ; last line "ZZ N1DM-READ-DRYCHECK PASS ZZ" or "... FAIL <n> ZZ".
H=F:/TanitAD-scratch/n1dc/read_n1dm
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
WC=F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py
export N1DM_TREE=F:/TanitAD-scratch/n1dc/build/n1dm_b1/tree CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 OMP_NUM_THREADS=1
export TMP=$H/tmp TEMP=$H/tmp
cd $H || exit 1
L=$H/dry_check_n1dm.log; : > $L; bad=0
run() { n=$1; shift; $PY -I $WC 5.5 60 >> $L 2>&1 || { echo "commit gate: $n" >> $L; bad=$((bad+1)); return; }
        "$@" >> $L 2>&1 && echo "PASS $n" >> $L || { echo "FAIL $n" >> $L; bad=$((bad+1)); }; }
for f in n1dm_ho_read.py n1dm_klim_run.py n1dm_t0route_lib.py n1dm_zpost.py read_lib.py replay_n1dh.py blobcheck.py test_n1dm_outcome.py test_n1dm_paths.py eval141/n1dm_dumps.py eval141/n1dm_read141.py eval141/n1dm_strategic.py eval141/n1dm_t0.py eval141/n1dm_view.py eval141/n1dw_view_s1slot.py eval141/k_hb0_141.py eval141/check_banked_n1d.py eval141/dry_check.py test/make_synth_ckpt.py; do
  run "py_compile $f" $PY -I -c "import sys; compile(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1], 'exec')" $f; done
for f in *.sh eval141/*.sh; do run "bash -n $f" bash -n $f; done
run "test_n1dm_outcome" $PY -I test_n1dm_outcome.py
run "test_n1dm_paths" $PY -I test_n1dm_paths.py
run "eval141/dry_check" $PY -I eval141/dry_check.py
[ $bad = 0 ] && echo "ZZ N1DM-READ-DRYCHECK PASS ZZ" >> $L || echo "ZZ N1DM-READ-DRYCHECK FAIL $bad ZZ" >> $L
