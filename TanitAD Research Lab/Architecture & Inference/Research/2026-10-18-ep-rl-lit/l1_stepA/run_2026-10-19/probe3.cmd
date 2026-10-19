@echo off
set L1_TREE=C:/Users/Admin/par18/l1run/tree_hybrid
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
cd /d C:\Users\Admin\par18\l1run
echo PROBE_START %DATE% %TIME% L1_TREE=%L1_TREE%
where nvidia-smi
C:\Users\Admin\venvs\tanitad\Scripts\python.exe -u C:\Users\Admin\par18\l1run\code\launch_stepA_ctype.py probe
> C:\Users\Admin\par18\l1run\PROBE3_RC echo %ERRORLEVEL%
echo PROBE_END
