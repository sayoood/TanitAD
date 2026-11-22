@echo off
rem run_n1dm_reads.cmd -- launch only through WMI (CreateFlags 0x08000200), stdin NUL. Git usr\bin first on PATH.
cd /d F:\TanitAD-scratch\n1dc\read_n1dm
set PATH=C:\Program Files\Git\usr\bin;%PATH%
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/n1dc/read_n1dm/run_n1dm_reads.sh < NUL
