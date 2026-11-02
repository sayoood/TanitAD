@echo off
rem SMOOTH-T1 K1 diagnostic launcher (WMI Win32_Process.Create).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/smooth_t1/k1_diag_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/smooth_t1/logs/k1_diag.launch.out 2>&1" < NUL
