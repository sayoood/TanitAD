@echo off
rem run_eval141_reboot.cmd -- relaunch after the 2026-11-20 02:33Z reboot. Same runner; gate 0 points at the MM release file. Launch only through WMI (CreateFlags 0x08000200), stdin NUL.
cd /d F:\TanitAD-scratch\n1dw\read\eval141
set PATH=C:\Program Files\Git\usr\bin;%PATH%
set N1DW_P5C_MARK=F:/TanitAD-scratch/n1dw/read/eval141/out/P5C_GATE_RELEASED_AFTER_REBOOT.txt
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/n1dw/read/eval141/run_eval141.sh < NUL
