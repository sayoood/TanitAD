@echo off
cd /d F:\TanitAD-scratch\pitch\code_v2
set PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
set CUDA_VISIBLE_DEVICES=-1
%PY% -I build.py render 1 2 3 4 7 8 --workers 1
set RC=%ERRORLEVEL%
%PY% -I build.py concat
echo render rc=%RC% concat rc=%ERRORLEVEL% > F:\TanitAD-scratch\pitch\out_v2\RENDER_DONE.txt
