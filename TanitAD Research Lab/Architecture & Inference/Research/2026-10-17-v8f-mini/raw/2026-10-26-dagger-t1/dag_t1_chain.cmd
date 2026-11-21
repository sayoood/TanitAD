@echo off
rem Detached launcher for dag_t1_chain.sh (start it through WMI Win32_Process.Create, as orc_t1_chain.cmd / hmc_chain.cmd). Arguments pass through (--dry-run).
rem Optional environment goes HERE (a WMI-created process does not inherit the caller's shell): DAGT1_TREE, DAGT1_B2_OUT, DAGT1_M2B_SHA_JSON, DAGT1_MAX_WAIT_H, DAGT1_READER, DAGT1_ART, DAGT1_OUT.
if not exist "E:\Projects\TanitAD-artifacts\dagt1\logs" mkdir "E:\Projects\TanitAD-artifacts\dagt1\logs"
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagt1/dag_t1_chain.sh %* < /dev/null" < NUL > E:\Projects\TanitAD-artifacts\dagt1\logs\dag_t1_chain.launch.out 2>&1
