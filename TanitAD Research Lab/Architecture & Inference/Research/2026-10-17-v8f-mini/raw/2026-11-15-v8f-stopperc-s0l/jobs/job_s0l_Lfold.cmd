@echo off
rem STOPPERC S0-L, BESIDE (Amendment 6): L_s0 T1-kin dumps on the FOLDED-weights eval cache, then STOPDEC on L_s0 into out/stopdec_Lfold. CPU only. Done marker: S0L_LFOLD.done.
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\s0l
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "CUDA_VISIBLE_DEVICES=-1"
set "STOPBAL_TREE=F:/TanitAD-scratch/stopbal_read/tree"
set "STOPBAL_DUMPS=F:/TanitAD-scratch/s0l_read/t1_fold"
set "S0L_DUMPS=F:/TanitAD-scratch/s0l_read/t1_fold"
set "S0L_L_CKPT=F:/TanitAD-scratch/s0l_read/L_s0/ckpt_last.pt"
set "S0L_L_SHA256=be8a63c8fdbdcf53f398f10839e73514869a783a70e59b96c12c29bec5887ab3"
set "S0L_L_CACHE=F:/TanitAD-scratch/s0l_read/cache_L_s0_fold"
if not exist F:\TanitAD-scratch\s0l_read\t1_fold mkdir F:\TanitAD-scratch\s0l_read\t1_fold
if exist S0L_LFOLD.done del S0L_LFOLD.done
echo [job] Lfold dumps start %DATE% %TIME% >> run_s0l_Lfold.log
%PY% -I -u run_dumps_s0l.py L_s0 >> run_s0l_Lfold.log 2>&1
set "RC1=%ERRORLEVEL%"
echo [job] Lfold dumps end %DATE% %TIME% rc=%RC1% >> run_s0l_Lfold.log
if not exist F:\TanitAD-scratch\s0l_read\t1_fold\done_L_s0_m10_causal_heads.txt (
  (echo dumps_rc=%RC1% stopdec=SKIPPED_no_done_file finished=%DATE% %TIME%)> S0L_LFOLD.done
  exit /b 3
)
echo [job] stopdec start %DATE% %TIME% >> run_s0l_Lfold.log
%PY% -I -u E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopbal\read\stopdec_arm.py --tags L_s0 --outdir E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\s0l\out\stopdec_Lfold --split-json >> run_s0l_Lfold.log 2>&1
set "RC2=%ERRORLEVEL%"
echo [job] stopdec end %DATE% %TIME% rc=%RC2% >> run_s0l_Lfold.log
(echo dumps_rc=%RC1% stopdec_rc=%RC2% finished=%DATE% %TIME%)> S0L_LFOLD.done
