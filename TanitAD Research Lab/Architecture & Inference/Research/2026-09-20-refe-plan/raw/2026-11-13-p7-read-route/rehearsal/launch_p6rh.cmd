@echo off
rem MM 2026-11-13: P7 read route families REHEARSAL on P6 (vs u400), tip 7f7ef7c7 export. CPU only.
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
cd /d C:\Users\Admin\p7rh
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" "C:\Users\Admin\p7rh\tip\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\eval\p7_read_chain.py" --ckpt F:/TanitAD_p6read_work/ckpt/run/model_final.pt --expect-ckpt-sha256 968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1 --name p6rh --base u400 --also p2 --data C:/Users/Admin/p7rh/data --only families --work C:/Users/Admin/p7rh/work > C:\Users\Admin\p7rh\chain_stdout.log 2>&1
echo ZZP6RH_EXIT %ERRORLEVEL% >> C:\Users\Admin\p7rh\chain_stdout.log
