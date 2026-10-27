@echo off
rem Detached launcher for dag2_t1_chain.sh (start it through WMI Win32_Process.Create, as dag_t1_chain.cmd). Arguments pass through (--dry-run).
rem Optional environment goes HERE (a WMI-created process does not inherit the caller shell): DAG2T1_TREE, DAG2T1_B2_OUT, DAG2T1_MAX_WAIT_H, DAG2T1_READER, DAG2T1_ART, DAG2T1_OUT, DAG2T1_C_MIN_MB, DAG2T1_E_MIN_MB, DAG2T1_E_ROLL_MB, DAG2T1_BANK_WAIT_H, DAG2T1_LEAD_BLOCK, DAG2T1_READA_CODE.
rem A real launch STOPS (ZZDAG2T1-STOP) until C: has >= 2048 MB free and E: >= 1500 MB free; it never deletes anything.
if not exist "E:\Projects\TanitAD-artifacts\dag2t1\logs" mkdir "E:\Projects\TanitAD-artifacts\dag2t1\logs"
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dag2t1/dag2_t1_chain.sh %* < /dev/null" < NUL > E:\Projects\TanitAD-artifacts\dag2t1\logs\dag2_t1_chain.launch.out 2>&1
