#!/usr/bin/env bash
K=/f/TanitAD-scratch/kinlab
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
"$PY" -I F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60 > $K/census_kinlab3.log 2>&1
export CUDA_VISIBLE_DEVICES=-1 TMP=F:/TanitAD-scratch/kinlab/tmp TEMP=F:/TanitAD-scratch/kinlab/tmp
"$PY" -I $K/census_kinlab2.py F:/TanitAD-scratch/kinlab/tmp/shard_000of004.npz F:/TanitAD-scratch/kinlab/tmp/native_tracks_union.pt F:/TanitAD-scratch/kinlab/KINLAB_CENSUS_3.json F:/TanitAD-scratch/kinlab/tree >> $K/census_kinlab3.log 2>&1
echo "CENSUS_RC $?" >> $K/census_kinlab3.log
