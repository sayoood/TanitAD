#!/bin/bash
# Control C7 rerun (C7b) with the PATCHED read_actsens.sh (windows + plan stages at torch threads 1, as phase 1): R35_s0 as the arm, banked R35_s0 TRAIN latents,
# HWM_WORKERS 2 (shared dev box: at most 2 plan processes). OUTROOT is c7diag/out so that no arm_*.json lands in the verdict's input folder.
# Marker: c7diag/out/C7b_R35_s0/ARM_DONE_C7b_R35_s0.txt or ARM_FAILED_*.txt; exit line in c7diag/out/c7b_run.exit; C7 verdict in c7diag/out/C7b_R35_s0/c7_result.json.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
export OUTROOT=F:/TanitAD-scratch/actsens/read/c7diag/out
source /f/TanitAD-scratch/actsens/read/env.sh
mkdir -p $OUTROOT
export ARM_C7=1 ARM_KEY=R35_s0 ARM_RUN_DIR=$PULLS/ladder_R35_s0 ARM_LATENTS=F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt HWM_WORKERS=2
bash /f/TanitAD-scratch/actsens/read/read_actsens.sh C7b_R35_s0 0
echo "c7b_run exit $? $(date -Iseconds)" >> $OUTROOT/c7b_run.exit
