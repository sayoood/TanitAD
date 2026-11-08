@echo off
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\capladder\env_capladder.cmd
set "LOG=run_capladder.run1.log"
set "RC=0"
set "RPT=n/a"
del CAPLADDER_RUN.done 2>nul
echo [job] start %DATE% %TIME% >> %LOG%
rem stage 1: the frame pull (ssh + tar reads only; light on RAM, so it runs before the memory gate)
%PY% -u pull_capladder.py >> %LOG% 2>&1
rem stage 2: the card's memory gate (>= 5,000 MB free)
%PY% -u gate_mem_capladder.py >> %LOG% 2>&1
if errorlevel 1 set "RC=2"
if not "%RC%"=="0" goto fin
%PY% -u sets_capladder.py >> %LOG% 2>&1
if errorlevel 1 set "RC=3"
if not "%RC%"=="0" goto fin
rem stage 3: R0 on the dev-box GPU (own status file; a crash or a held GPU is NOT RUN with the reason, the rest still runs)
set "CUDA_VISIBLE_DEVICES="
%PY% -u r0_capladder.py >> %LOG% 2>&1
set "CUDA_VISIBLE_DEVICES=-1"
rem stage 4: R1 / R2 / R2m / R3 / R4 features (CPU)
%PY% -u feats_capladder.py >> %LOG% 2>&1
if errorlevel 1 set "RC=4"
if not "%RC%"=="0" goto fin
rem stage 5: Grams (GPU when free) + 3 workers
set "CUDA_VISIBLE_DEVICES="
%PY% -u run_capladder.py >> %LOG% 2>&1
set "RC=%ERRORLEVEL%"
set "CUDA_VISIBLE_DEVICES=-1"
if not "%RC%"=="0" goto fin
%PY% -u report_capladder.py >> %LOG% 2>&1
set "RPT=%ERRORLEVEL%"
%PY% -u figs_capladder.py >> %LOG% 2>&1
%PY% -u report_capladder.py >> %LOG% 2>&1
%PY% -u cleanup_capladder.py >> %LOG% 2>&1
:fin
(echo rc=%RC%)> CAPLADDER_RUN.done
(echo report=%RPT%)>> CAPLADDER_RUN.done
echo [job] end %DATE% %TIME% rc=%RC% >> %LOG%
