#!/bin/bash
# usage: one_unit.sh <label> <tree_root> <MKL_NUM_THREADS> <chassis> <arm> <seed> <batch> [c7_plans_dir]
# OMP_NUM_THREADS is 2 in every experiment (phase 1 and C7 both exported 2); MKL_NUM_THREADS is the variable under test
# (torch reads MKL_NUM_THREADS after OMP_NUM_THREADS for its intra-op default, so MKL decides torch.get_num_threads()).
export PATH="/usr/bin:/mingw64/bin:/bin:$PATH"
L=$1; T=$2; M=$3; K=$4; ARM=$5; SEED=$6; B=$7; C7=${8:-}
D=F:/TanitAD-scratch/actsens/read/c7diag
mkdir -p $D/tmp
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=$M PYTHONDONTWRITEBYTECODE=1 WMACT_CODE_ROOT=$T LADDER_READ_CODE_ROOT=$T HF_HUB_DISABLE_XET=1
export PYTHONPATH="$(cygpath -w $T/stack);$(cygpath -w $T/taniteval)"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
cd E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/hwm_plan
$PY -u $D/one_unit.py $L $K $ARM $SEED $B $D/tmp/$L.npz $C7 </dev/null > $D/tmp/$L.log 2>&1
echo "rc=$? $(date -Iseconds)" > $D/tmp/$L.done
