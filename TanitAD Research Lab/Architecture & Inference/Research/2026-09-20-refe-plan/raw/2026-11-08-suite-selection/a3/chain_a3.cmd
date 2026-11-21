@echo off
REM bundle_a3 chain: prepare -> (a0 forward on CPU, concurrent) + extract -> sky -> wait for a0 -> assemble.  Launch detached: cmd /c chain_a3.cmd < NUL
set CUDA_VISIBLE_DEVICES=-1
set PYTHONDONTWRITEBYTECODE=1
set PY=C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe
set FD=F:\TanitAD_tsf_bundle2_20261107
set LOG=%FD%\work_build_a3_chain.log
cd /d E:\Projects\TanitAD-artifacts\tsf_bundle2_20261107\a3
if not exist %FD%\work_a3 mkdir %FD%\work_a3
echo [%time%] prepare > %LOG%
%PY% -I build_bundle_a3.py prepare >> %LOG% 2>&1
echo [%time%] a0 forward start >> %LOG%
start "" /b cmd /c "%PY% -I E:\Projects\TanitAD-artifacts\tsf_bundle2_20261107\code\refe_a0_inputs_b2.py --tokens %FD%\work_a3\wanted.json --out %FD%\a0_forward_a3 --device cpu --forward > %FD%\a0_forward_a3.out 2>&1"
echo [%time%] extract >> %LOG%
%PY% -I build_bundle_a3.py extract 6 >> %LOG% 2>&1
echo [%time%] sky >> %LOG%
%PY% -I build_bundle_a3.py sky >> %LOG% 2>&1
:wait
if exist %FD%\a0_forward_a3\DONE goto go
ping -n 21 127.0.0.1 > nul
goto wait
:go
echo [%time%] assemble >> %LOG%
%PY% -I build_bundle_a3.py assemble >> %LOG% 2>&1
echo [%time%] CHAIN_DONE >> %LOG%
