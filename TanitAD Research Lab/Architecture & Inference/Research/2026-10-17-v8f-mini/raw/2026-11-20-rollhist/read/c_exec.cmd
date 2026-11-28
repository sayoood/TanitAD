@echo off
rem c_exec.cmd -- control c-EXEC, launched by WMI (CreateFlags 0x08000200), stdin NUL.
cd /d F:\TanitAD-scratch\rollhist\read
set PATH=C:\Program Files\Git\usr\bin;%PATH%
set TMP=F:\TanitAD-scratch\rollhist\read\tmp
set TEMP=F:\TanitAD-scratch\rollhist\read\tmp
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" -u F:\TanitAD-scratch\rollhist\read\c_exec.py > F:\TanitAD-scratch\rollhist\read\out\c_exec.log 2>&1 < NUL
