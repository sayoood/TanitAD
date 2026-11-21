@echo off
rem STOPPERC stage-1 read, step 3: STOPDEC (T's goal / lon probabilities, the strata; first tick) of the stage-1 arms, CPU, zero training. Run after DUMPS_S1.done reads rc=0.
rem The instrument is the first dose's stopdec_arm.py, unchanged; the tree and the dump folder move by env (STOPBAL_TREE, STOPBAL_DUMPS). The reused BAL-NULL arms keep their first-dose STOPDEC files.
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\stage1_A\read
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "CUDA_VISIBLE_DEVICES=-1"
set "STOPBAL_TREE=F:/TanitAD-scratch/stopperc_read/tree"
set "STOPBAL_DUMPS=F:/TanitAD-scratch/stopperc_read/t1"
if exist STOPDEC_S1.done del STOPDEC_S1.done
set "TAGS="
for /f "delims=" %%T in ('%PY% -I stopperc_s1_read.py tags') do set "TAGS=%%T"
if "%TAGS%"=="" (
  echo [job] REFUSED: no tags (the NULL reuse state is not clean) >> run_stopdec_s1.log
  (echo rc=2 finished=%DATE% %TIME%)> STOPDEC_S1.done
  exit /b 2
)
echo [job] start %DATE% %TIME% tags=%TAGS% >> run_stopdec_s1.log
%PY% -I -u E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopbal\read\stopdec_arm.py --tags %TAGS% --outdir E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\stage1_A\read\out\stopdec --split-json >> run_stopdec_s1.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_stopdec_s1.log
(echo rc=%RC% finished=%DATE% %TIME%)> STOPDEC_S1.done
