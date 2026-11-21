@echo off
rem S0 job (PREREG_V8F_PROOF_ADD12 Stage 0): CPU only, no GPU, no lock. Waits for free commit >= 6 GB, then runs the full read on the TRUE labels. Launch via WMI, not from an agent shell with a pipe.
set "PATH=C:\Windows\System32;C:\Windows;C:\Windows\System32\WindowsPowerShell\v1.0"
set PYTHONIOENCODING=utf-8
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
cd /d F:\TanitAD-scratch\v8proof\spatial\s0
if exist S0_DONE.txt del S0_DONE.txt
if exist S0_STOP.txt del S0_STOP.txt
set "WAITS=0"
:waitmem
set "FREE=0"
for /f %%i in ('powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB)"') do set "FREE=%%i"
if %FREE% GEQ 6 goto memok
set /a WAITS+=1
if %WAITS% GEQ 240 ( echo ZZ S0-STOP commit-wait-timeout free=%FREE%GB > S0_STOP.txt & exit /b 1 )
echo wait %WAITS% free %FREE% GB >> s0_job.log
ping -n 121 127.0.0.1 > NUL
goto waitmem
:memok
echo start free %FREE% GB %DATE% %TIME% >> s0_job.log
C:\Users\Admin\venvs\tanitad\Scripts\python.exe s0_read.py run >> s0_job.log 2>&1
set "RC=%ERRORLEVEL%"
if "%RC%"=="2" ( echo ZZ S0-STOP control-missed rc 2 see VERDICT_S0.md and s0_job.log > S0_STOP.txt & exit /b 2 )
if not "%RC%"=="0" ( echo ZZ S0-STOP run-failed rc %RC% see s0_job.log > S0_STOP.txt & exit /b 1 )
echo ZZ S0-DONE %DATE% %TIME% > S0_DONE.txt
exit /b 0
