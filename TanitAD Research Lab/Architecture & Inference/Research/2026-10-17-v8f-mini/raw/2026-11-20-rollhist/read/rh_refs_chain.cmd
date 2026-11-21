@echo off
rem rh_refs_chain.cmd -- launch only through WMI (CreateFlags 0x08000200), stdin NUL.
cd /d F:\TanitAD-scratch\rollhist\read
set PATH=C:\Program Files\Git\usr\bin;%PATH%
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/rollhist/read/rh_refs_chain.sh < NUL
