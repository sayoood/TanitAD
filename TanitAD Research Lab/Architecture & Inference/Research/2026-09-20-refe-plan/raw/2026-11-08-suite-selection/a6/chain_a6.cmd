@echo off
REM bundle_a6 chain: prepare -> (a0 forward on CPU, concurrent) + extract -> sky -> wait for a0 -> assemble.  Launch detached: cmd /c chain_a6.cmd < NUL
set CUDA_VISIBLE_DEVICES=-1
set PYTHONDONTWRITEBYTECODE=1
set PY=C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe
set FD=F:\TanitAD_tsf_bundle2_20261107
set LOG=%FD%\work_build_a6_chain.log
cd /d E:\Projects\TanitAD-artifacts\tsf_bundle2_20261107\a6
if not exist %FD%\work_a6 mkdir %FD%\work_a6
echo [%time%] prepare > %LOG%
%PY% -I build_bundle_a6.py prepare >> %LOG% 2>&1
echo [%time%] a0 forward start >> %LOG%
start "" /b cmd /c "%PY% -I E:\Projects\TanitAD-artifacts\tsf_bundle2_20261107\code\refe_a0_inputs_b2.py --tokens %FD%\work_a6\wanted.json --out %FD%\a0_forward_a6 --device cpu --forward > %FD%\a0_forward_a6.out 2>&1"
echo [%time%] extract >> %LOG%
%PY% -I build_bundle_a6.py extract 6 >> %LOG% 2>&1
echo [%time%] sky >> %LOG%
%PY% -I build_bundle_a6.py sky >> %LOG% 2>&1
:wait
if exist %FD%\a0_forward_a6\DONE goto go
ping -n 21 127.0.0.1 > nul
goto wait
:go
echo [%time%] assemble >> %LOG%
%PY% -I build_bundle_a6.py assemble >> %LOG% 2>&1
if errorlevel 1 (
  echo [%time%] CHAIN_FAIL assemble >> %LOG%
  exit /b 1
)
echo [%time%] CHAIN_DONE >> %LOG%
