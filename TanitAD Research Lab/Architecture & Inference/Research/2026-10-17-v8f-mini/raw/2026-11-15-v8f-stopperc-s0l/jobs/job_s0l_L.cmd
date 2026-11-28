@echo off
rem STOPPERC S0-L steps 4-5 for L_s0 (MM 2026-11-15): T1-kin dumps of L_s0 on its OWN L-merged eval cache (run_dumps_s0l.py, same tree and threads as BPRIME_s0's 11-13 dumps),
rem then STOPDEC on BOTH tags into out/stopdec_L (one environment). CPU only, zero training. Done marker: S0L_L.done (rc of each step).
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\s0l
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "CUDA_VISIBLE_DEVICES=-1"
set "STOPBAL_TREE=F:/TanitAD-scratch/stopbal_read/tree"
set "STOPBAL_DUMPS=F:/TanitAD-scratch/s0l_read/t1"
set "S0L_DUMPS=F:/TanitAD-scratch/s0l_read/t1"
set "S0L_L_CKPT=F:/TanitAD-scratch/s0l_read/L_s0/ckpt_last.pt"
set "S0L_L_SHA256=be8a63c8fdbdcf53f398f10839e73514869a783a70e59b96c12c29bec5887ab3"
set "S0L_L_CACHE=F:/TanitAD-scratch/s0l_read/cache_L_s0"
if exist S0L_L.done del S0L_L.done
echo [job] L dumps start %DATE% %TIME% >> run_s0l_L.log
%PY% -I -u run_dumps_s0l.py L_s0 >> run_s0l_L.log 2>&1
set "RC1=%ERRORLEVEL%"
echo [job] L dumps end %DATE% %TIME% rc=%RC1% >> run_s0l_L.log
if not exist F:\TanitAD-scratch\s0l_read\t1\done_L_s0_m10_causal_heads.txt (
  (echo dumps_rc=%RC1% stopdec=SKIPPED_no_done_file finished=%DATE% %TIME%)> S0L_L.done
  exit /b 3
)
echo [job] stopdec start %DATE% %TIME% >> run_s0l_L.log
%PY% -I -u E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopbal\read\stopdec_arm.py --tags BPRIME_s0 L_s0 --outdir E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\s0l\out\stopdec_L --split-json >> run_s0l_L.log 2>&1
set "RC2=%ERRORLEVEL%"
echo [job] stopdec end %DATE% %TIME% rc=%RC2% >> run_s0l_L.log
(echo dumps_rc=%RC1% stopdec_rc=%RC2% finished=%DATE% %TIME%)> S0L_L.done
