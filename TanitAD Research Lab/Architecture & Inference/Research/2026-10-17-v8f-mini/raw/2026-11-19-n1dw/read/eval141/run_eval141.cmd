@echo off
rem run_eval141.cmd -- launch only through WMI (CreateFlags 0x08000200), stdin NUL. Git usr\bin goes first on PATH (a bare WMI environment lacks it); usr\bin\bash.exe runs the script directly.
cd /d F:\TanitAD-scratch\n1dw\read\eval141
set PATH=C:\Program Files\Git\usr\bin;%PATH%
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/n1dw/read/eval141/run_eval141.sh < NUL
