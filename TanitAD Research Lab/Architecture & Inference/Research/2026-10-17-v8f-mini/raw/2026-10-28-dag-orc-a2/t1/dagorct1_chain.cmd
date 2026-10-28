@echo off
rem Detached launcher for dagorct1_chain.sh (start it through WMI Win32_Process.Create so it survives the caller; arguments pass through, e.g. --dry-run).
rem   powershell -NoProfile -Command "([wmiclass]'Win32_Process').Create('cmd /c E:\Projects\TanitAD-artifacts\dagorc_t1\dagorct1_chain.cmd') | Select-Object ReturnValue, ProcessId"
rem RESUME = launch it again (finished steps are skipped by their done records).  Output: logs\chain.out (the chain's own log, anchored markers) and logs\dagorct1_chain.launch.out (launcher stdout).
rem Optional environment goes HERE (a WMI-created process does not inherit the caller shell), e.g.:
rem   set DORT1_OUT=F:/TanitAD-scratch/dagorc_t1_out         (if E: is too full: E: had ~2.6 GB free for ~1.7 GB of new data plus the box-2 pulls' 1 GiB floor)
rem   set DORT1_ART=E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc   (must equal box 2's DAGOACCORC_ART)   set DORT1_MAX_WAIT_H=12
if not exist "E:\Projects\TanitAD-artifacts\dagorc_t1\logs" mkdir "E:\Projects\TanitAD-artifacts\dagorc_t1\logs"
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagorc_t1/dagorct1_chain.sh %* < /dev/null >> /e/Projects/TanitAD-artifacts/dagorc_t1/logs/dagorct1_chain.launch.out 2>&1" < NUL
