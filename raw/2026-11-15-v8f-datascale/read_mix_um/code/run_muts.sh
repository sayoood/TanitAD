export PATH="/usr/bin:/bin:$PATH" CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
cd F:/TanitAD-scratch/datascale/read_mix
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -u mutate_check_mix.py > out/mutate_mix_run.txt 2>&1
C:/Users/Admin/venvs/tanitad/Scripts/python.exe -u mutate_check_mix.py --smoke > out/mutate_mix_smoke_run.txt 2>&1
echo done > out/muts_done.txt
