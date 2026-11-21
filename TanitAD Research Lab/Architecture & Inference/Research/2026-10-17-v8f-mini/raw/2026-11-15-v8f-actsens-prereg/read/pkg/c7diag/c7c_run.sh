#!/bin/bash
# Control C7 on chassis s1 (C7c, MM 2026-11-15): the PATCHED read_actsens.sh (torch threads 1 for windows + plans, as phase 1) with R35_s1 itself as the arm,
# banked R35_s1 TRAIN latents, HWM_WORKERS 2. OUTROOT c7diag/out (no arm_*.json lands in the verdict's input folder).
# Marker: c7diag/out/C7c_R35_s1/ARM_DONE_C7c_R35_s1.txt or ARM_FAILED_*.txt; exit line c7diag/out/c7c_run.exit; verdict c7diag/out/C7c_R35_s1/c7_result.json.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
export OUTROOT=F:/TanitAD-scratch/actsens/read/c7diag/out
source /f/TanitAD-scratch/actsens/read/env.sh
mkdir -p $OUTROOT
export ARM_C7=1 ARM_KEY=R35_s1 ARM_RUN_DIR=$PULLS/ladder_R35_s1 ARM_LATENTS=F:/TanitAD-scratch/hwm_plan/latents_train_R35_s1.pt HWM_WORKERS=2
bash /f/TanitAD-scratch/actsens/read/read_actsens.sh C7c_R35_s1 1
echo "c7c_run exit $? $(date -Iseconds)" >> $OUTROOT/c7c_run.exit
