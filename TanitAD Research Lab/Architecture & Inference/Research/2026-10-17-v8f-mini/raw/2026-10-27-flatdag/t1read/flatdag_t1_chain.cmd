@echo off
rem Detached launcher for flatdag_t1_chain.sh (start it through WMI Win32_Process.Create, as dag2_t1_chain.cmd). Arguments pass through (--dry-run).
rem Optional environment goes HERE (a WMI-created process does not inherit the caller shell): FDT1_TREE, FDT1_B2_OUT (default C:/Users/Admin/box2_ship/logs/chain_v8f_flatdag.out; "off" skips), FDT1_MAX_WAIT_H, FDT1_READER, FDT1_ART, FDT1_OUT, FDT1_C_MIN_MB, FDT1_E_MIN_MB, FDT1_E_ROLL_MB, FDT1_LEAD_BLOCK, FDT1_READA_CODE.
rem A real launch STOPS (ZZFLATDAGT1-STOP) until C: has >= 2048 MB free and E: >= 1500 MB free; it never deletes anything.
if not exist "E:\Projects\TanitAD-artifacts\flatdag_t1\logs" mkdir "E:\Projects\TanitAD-artifacts\flatdag_t1\logs"
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/flatdag_t1/flatdag_t1_chain.sh %* < /dev/null" < NUL > E:\Projects\TanitAD-artifacts\flatdag_t1\logs\flatdag_t1_chain.launch.out 2>&1
