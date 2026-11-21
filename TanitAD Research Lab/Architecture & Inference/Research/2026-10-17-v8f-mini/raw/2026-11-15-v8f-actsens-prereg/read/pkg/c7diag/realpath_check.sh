#!/bin/bash
# Real-path check of the patched driver's plan command: env.sh, then the exact step-4 command line of read_actsens.sh (MKL 1, --workers 2, so the spawn-pool path),
# on the first 128 SCORE windows of R35_s0 (batches 0 and 1; batch j keeps its generator make_gen(seed, j)), hwm_oracle seed 1. Output in c7diag/tmp/rp.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
source /f/TanitAD-scratch/actsens/read/env.sh
R=F:/TanitAD-scratch/actsens/read/c7diag/tmp/rp
KEY=R35_s0; RUN_DIR=$PULLS/ladder_R35_s0; LAT=F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt; W=2
cd $H
OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 $PY -c "import torch,sys; n=torch.get_num_threads(); print('plan env torch threads', n); sys.exit(0 if n == 1 else 1)" </dev/null > $R/rp.log 2>&1 || { echo "rc=9 thread probe" > $R/rp.done; exit 9; }
COM="--tag $KEY --ckpt $RUN_DIR/ckpt_last.pt --train-latents $LAT --train-tag $KEY --seed-mode prior --lam 0 --out $R"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 $PY -u run_hwm.py plan $COM --arms hwm_oracle --seeds 1 --workers $W </dev/null >> $R/rp.log 2>&1
echo "rc=$? $(date -Iseconds)" > $R/rp.done
