@echo off
rem STOPPERC S0-L prep: B-prime s0 T1-kin dumps (run_dumps_s0l.py) then STOPDEC on BPRIME_s0 only. CPU, zero training, no GPU. Done marker: S0L_BPRIME.done (rc of each step).
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\s0l
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "CUDA_VISIBLE_DEVICES=-1"
set "STOPBAL_TREE=F:/TanitAD-scratch/stopbal_read/tree"
set "STOPBAL_DUMPS=F:/TanitAD-scratch/s0l_read/t1"
set "S0L_DUMPS=F:/TanitAD-scratch/s0l_read/t1"
if exist S0L_BPRIME.done del S0L_BPRIME.done
echo [job] dumps start %DATE% %TIME% >> run_s0l_bprime.log
%PY% -I -u run_dumps_s0l.py BPRIME_s0 >> run_s0l_bprime.log 2>&1
set "RC1=%ERRORLEVEL%"
echo [job] dumps end %DATE% %TIME% rc=%RC1% >> run_s0l_bprime.log
if not exist F:\TanitAD-scratch\s0l_read\t1\done_BPRIME_s0_m10_causal_heads.txt (
  (echo dumps_rc=%RC1% stopdec=SKIPPED_no_done_file finished=%DATE% %TIME%)> S0L_BPRIME.done
  exit /b 3
)
echo [job] stopdec start %DATE% %TIME% >> run_s0l_bprime.log
%PY% -I -u E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopbal\read\stopdec_arm.py --tags BPRIME_s0 --outdir E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\s0l\out\stopdec --split-json >> run_s0l_bprime.log 2>&1
set "RC2=%ERRORLEVEL%"
echo [job] stopdec end %DATE% %TIME% rc=%RC2% >> run_s0l_bprime.log
(echo dumps_rc=%RC1% stopdec_rc=%RC2% finished=%DATE% %TIME%)> S0L_BPRIME.done
