@echo off
rem rh_read_ra_retry.cmd -- MM 2026-11-22. Launch only through WMI (CreateFlags 0x08000200), stdin NUL. Runs rh_read_ra_retry.sh, which re-runs rh_read_ra_chain.sh only on a commit-headroom stop.
cd /d F:\TanitAD-scratch\rollhist\read
set PATH=C:\Program Files\Git\usr\bin;%PATH%
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/rollhist/read/rh_read_ra_retry.sh < NUL
