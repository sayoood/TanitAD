export PATH="/usr/bin:/bin:$PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 DS_BLOCK=mix
TREE=F:/TanitAD-scratch/otband_read/tree
cd F:/TanitAD-scratch/datascale/read_mix
export PYTHONPATH="$(cygpath -w $TREE/stack);$(cygpath -w $TREE/taniteval)"
for w in bl ue; do C:/Users/Admin/venvs/tanitad/Scripts/python.exe -u smoke_ds_read_mix.py $w > out/smoke_mix_$w.txt 2>&1; done
echo done > out/smoke_mix_done.txt
