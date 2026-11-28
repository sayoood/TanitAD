#!/usr/bin/env bash
K=/f/TanitAD-scratch/kinlab
bash $K/run_tests.sh kc_final_kincache.log $K/tree2 stack/tests/test_v8f_kincache.py
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
"$PY" -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 > $K/kc_real.log 2>&1
CUDA_VISIBLE_DEVICES=-1 TMP=F:/TanitAD-scratch/kinlab/tmp TEMP=F:/TanitAD-scratch/kinlab/tmp "$PY" -I $K/kincache_real.py F:/TanitAD-scratch/kinlab/tmp/shard_000of004.npz F:/TanitAD-scratch/kinlab/tmp/native_tracks_union.pt $K/KINCACHE_REAL.json F:/TanitAD-scratch/kinlab/tree2 F:/TanitAD-scratch/kinlab/tmp/kcreal 4000 >> $K/kc_real.log 2>&1
echo "RC $?" >> $K/kc_real.log
echo FINAL_DONE > $K/kc_final_done.txt
