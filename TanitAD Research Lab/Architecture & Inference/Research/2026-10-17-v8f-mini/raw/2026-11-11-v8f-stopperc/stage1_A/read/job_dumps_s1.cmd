@echo off
rem STOPPERC stage-1 read, step 2: T1-kin dumps (CPU, zero training, NO GPU: CUDA_VISIBLE_DEVICES=-1, so no devbox_gpu.lock is taken, as in STOPBAL's job_dumps.cmd). Needs F:\TanitAD-scratch\stopperc_read\tree (make_tree_s1.py).
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\stage1_A\read
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "CUDA_VISIBLE_DEVICES=-1"
if exist DUMPS_S1.done del DUMPS_S1.done
echo [job] start %DATE% %TIME% >> run_dumps_s1.log
%PY% -I -u stopperc_s1_read.py dumps >> run_dumps_s1.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_dumps_s1.log
(echo rc=%RC% finished=%DATE% %TIME%)> DUMPS_S1.done
