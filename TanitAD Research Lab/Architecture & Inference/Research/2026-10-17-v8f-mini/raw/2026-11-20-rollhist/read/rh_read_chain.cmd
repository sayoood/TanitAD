@echo off
rem rh_read_chain.cmd -- launch only through WMI (CreateFlags 0x08000200), stdin NUL. Git usr\bin goes first on PATH; usr\bin\bash.exe runs the script directly.
cd /d F:\TanitAD-scratch\rollhist\read
set PATH=C:\Program Files\Git\usr\bin;%PATH%
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/rollhist/read/rh_read_chain.sh < NUL
