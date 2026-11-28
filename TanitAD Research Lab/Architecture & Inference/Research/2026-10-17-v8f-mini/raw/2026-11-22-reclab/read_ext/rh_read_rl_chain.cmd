@echo off
rem rh_read_rl_chain.cmd -- launch only through WMI (CreateFlags 0x08000200), stdin NUL. Git usr\bin goes first on PATH; usr\bin\bash.exe runs the script directly. Launch only after the pod arms RL_s0 / RL_s1 show ARM_DONE.
cd /d F:\TanitAD-scratch\rollhist\read
set PATH=C:\Program Files\Git\usr\bin;%PATH%
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/rollhist/read/rh_read_rl_chain.sh < NUL
