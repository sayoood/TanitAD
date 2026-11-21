#!/bin/bash
# Environment of the ACTSENS read (dev-box CPU). Sourced by read_actsens.sh. Tier T0. The ACTSENS tree is the code root for every Python process (asserted by wmact_lib, lr_lib and check_tree.py).
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${OMP_NUM_THREADS:-2}" PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1
export RD=F:/TanitAD-scratch/actsens/read
export ACT_TREE=$RD/tree
export WMACT_CODE_ROOT=$ACT_TREE LADDER_READ_CODE_ROOT=$ACT_TREE
export PYTHONPATH="$(cygpath -w $ACT_TREE/stack);$(cygpath -w $ACT_TREE/taniteval)"
export PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export THEORY=E:/Projects/TanitAD-artifacts/dagland/motion_c/theory
export H=$THEORY/hwm_plan
export HWMTOOLS=F:/TanitAD-scratch/hwm_plan
export PULLS=C:/Users/Admin/t2v8f_pulls
export R35REF=$RD/hf_phase1/R35/plan_runs
export OUTROOT=${OUTROOT:-$RD/out}
export ACT_COMMIT=6b2bca1fbd4fb9c930c67989b7ba1c599ef8d090
