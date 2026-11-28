@echo off
set OMP_NUM_THREADS=2
set CUDA_VISIBLE_DEVICES=-1
set PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
set K=C:/Users/Admin/par18/epadmc1/repo/2026-10-19-p6-read-prep/code
set BK=C:/Users/Admin/par18/epadmc1/work/c1_bank
set OUT=C:/Users/Admin/par18/epadmc1/work/gate
echo START %DATE% %TIME% > %OUT%/gate_started.txt
cd /d C:/Users/Admin/par18/epadmc1/repo/2026-10-19-p6-read-prep/code
%PY% -I ep_admission.py run --rules r1 --p2-bank %BK%/p2 --cand-bank %BK%/s0 --cand-bank %BK%/s1 --cand-bank %BK%/s2 --cand-id s0 --cand-id s1 --cand-id s2 --ranked-logs %BK%/ranked_logs.tsv --no-ranked-pin --allow-unsealed --out %OUT%/admission_c1.json
set RC=%ERRORLEVEL%
echo gate_rc=%RC% %DATE% %TIME% > %OUT%/gate_rc.txt
if "%RC%"=="0" goto margin
if "%RC%"=="3" goto margin
goto done
:margin
cd /d C:/Users/Admin/par18/epadmc1/repo/2026-10-19-epadm-searchbank/code
%PY% -I apply_a2_margin.py %OUT%/admission_c1.json --margin 1.05 --out %OUT%/admission_c1_margin105.json
:done
echo DONE %DATE% %TIME% > %OUT%/GATE_DONE.marker
