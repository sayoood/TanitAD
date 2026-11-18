M=F:/TanitAD-scratch/datascale/mix_impl
T=$M/tree
export PYTHONPATH="$(cygpath -w $T/stack);$(cygpath -w $T)"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
mkdir -p $M/tmp/run $M/logs
