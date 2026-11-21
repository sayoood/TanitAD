@echo off
rem Detached launcher for dagoacc_t1_chain.sh (start it through WMI Win32_Process.Create, as flatdag_t1_chain.cmd). Arguments pass through (--dry-run).
rem Optional environment goes HERE (a WMI-created process does not inherit the caller shell): DOT1_TREE, DOT1_B2_OUT (default C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out; "off" skips), DOT1_MAX_WAIT_H, DOT1_READER, DOT1_ART, DOT1_OUT, DOT1_C_MIN_MB, DOT1_E_MIN_MB, DOT1_E_ROLL_MB, DOT1_LEAD_BLOCK, DOT1_READA_CODE.
rem A real launch STOPS (ZZDAGOACCT1-STOP) until C: has >= 2048 MB free and E: >= 1500 MB free; it never deletes anything.
if not exist "E:\Projects\TanitAD-artifacts\dagoacc_t1\logs" mkdir "E:\Projects\TanitAD-artifacts\dagoacc_t1\logs"
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagoacc_t1/dagoacc_t1_chain.sh %* < /dev/null" < NUL > E:\Projects\TanitAD-artifacts\dagoacc_t1\logs\dagoacc_t1_chain.launch.out 2>&1
