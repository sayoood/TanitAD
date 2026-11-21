#!/bin/bash
# Control COLREPRO (MM 2026-11-15): does the read package's latent collector (collect_latents_arm.py, run as read_actsens.sh step 3 runs it)
# reproduce phase 1's banked R35_s0 TRAIN latents (F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt) exactly?
# Needed before any ACTSENS arm is read: the arm's own latents come from this collector, R35's from phase 1's. Read-only sftp from box 2. CPU only.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
source /f/TanitAD-scratch/actsens/read/env.sh
OUT=F:/TanitAD-scratch/actsens/read/c7diag/out/COLREPRO_R35_s0
mkdir -p $OUT/ts/cache/train/clips; cp -n $RD/ts/clips141.json $OUT/ts/
echo "$(date -Iseconds) collect start" >> $OUT/colrepro.log
COL_OUT_DIR=$OUT COL_TAG=R35_s0 COL_TS=$OUT/ts COL_CKPT=$PULLS/ladder_R35_s0/ckpt_last.pt OMP_NUM_THREADS=4 $PY -u $RD/collect_latents_arm.py </dev/null >> $OUT/collect.log 2>&1
echo "$(date -Iseconds) collect rc $?" >> $OUT/colrepro.log
$PY -u $RD/c7diag/colrepro_cmp.py $OUT/latents_train_R35_s0.pt F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt $OUT/colrepro_result.json </dev/null >> $OUT/colrepro.log 2>&1
echo "$(date -Iseconds) compare rc $?" >> $OUT/colrepro.log
