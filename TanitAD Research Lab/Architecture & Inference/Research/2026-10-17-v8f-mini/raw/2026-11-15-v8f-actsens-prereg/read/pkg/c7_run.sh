#!/bin/bash
# Control C7 (MEASURED): the ACTSENS driver with R35_s0 itself as the arm, on the banked R35_s0 TRAIN latents. Detached; marker out/C7_R35_s0/ARM_DONE_C7_R35_s0.txt or ARM_FAILED_*.txt.
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
source /f/TanitAD-scratch/actsens/read/env.sh
export ARM_C7=1 ARM_KEY=R35_s0 ARM_RUN_DIR=$PULLS/ladder_R35_s0 ARM_LATENTS=F:/TanitAD-scratch/hwm_plan/latents_train_R35_s0.pt HWM_WORKERS=${HWM_WORKERS:-3}
bash /f/TanitAD-scratch/actsens/read/read_actsens.sh C7_R35_s0 0
echo "c7_run exit $? $(date -Iseconds)" >> /f/TanitAD-scratch/actsens/read/out/C7_run.exit
