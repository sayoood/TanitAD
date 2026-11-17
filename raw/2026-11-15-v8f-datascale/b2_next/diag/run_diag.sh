#!/bin/bash
# run_diag.sh -- DSCALE10K diagnostics, CPU only, one heavy job. Sentinel: "ZZ DIAG DONE ZZ" / "ZZ DIAG FAIL ZZ" in out/diag.log
export PATH="/usr/bin:/bin:$PATH"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 DS_BLOCK=2
D=F:/TanitAD-scratch/datascale/diag
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p $D/out
cd $D || exit 1
echo "[diag] start $(date -u +%FT%TZ)" > $D/out/diag.log
$PY F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 6 120 >> $D/out/diag.log 2>&1 || { echo "ZZ DIAG FAIL ZZ commit" >> $D/out/diag.log; exit 1; }
$PY $D/diag_read.py --part both --n-boot 2000 --out $D/RESULT_DIAG.json >> $D/out/diag.log 2>&1 && echo "ZZ DIAG DONE ZZ" >> $D/out/diag.log || echo "ZZ DIAG FAIL ZZ rc=$?" >> $D/out/diag.log
