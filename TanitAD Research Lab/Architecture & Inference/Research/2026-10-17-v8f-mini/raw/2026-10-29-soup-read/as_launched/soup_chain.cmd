@echo off
rem Detached launcher for soup_chain.sh, the dev-box CPU chain of PREREG_V8F_SOUP section 13a (start it through WMI Win32_Process.Create so it survives the caller; arguments pass through, e.g. --dry-run).
rem   powershell -NoProfile -Command "([wmiclass]'Win32_Process').Create('cmd /c E:\Projects\TanitAD-artifacts\dagland\soup\soup_chain.cmd') | Select-Object ReturnValue, ProcessId"
rem RESUME = launch it again (a finished step is skipped by its done record; an incomplete earlier attempt is MOVED to <OUT>\_incomplete_*, never deleted).
rem Output: logs\chain.out (the chain's own log, anchored ZZSOUP-* markers; rotated at every start) and logs\soup_chain.launch.out (the launcher's stdout; truncated at every launch).  Wait for it with: bash wake_soup.sh
rem Optional environment goes HERE (a WMI-created process does not inherit the caller shell).  The defaults are the production values (see the header of soup_chain.sh).
rem
rem REQUIRED, and left UNSET as shipped so the chain REFUSES to start (it prints ZZSOUP-STOP and runs nothing):  the promoted PREREG_V8F_SOUP and its sha256, which become the reader's --prereg and --prereg-sha.
rem After the PREREG is promoted and landed: delete the leading "rem   " of the next two lines and fill them in (no quotes; the file must hash to the sha256).
set SOUP_PREREG=F:/TanitAD-scratch/soup_prereg/PREREG_V8F_SOUP.md
set SOUP_PREREG_SHA=830d9c9143850628188f7d8489566e92f2a9cbdf050689331f1eaa7d3c78993c
rem
rem The RAM guard is the LAUNCH-INTENT chain's 4096 MB lowered to 2048 MB (Master Mind 2026-10-28 15:12: the PI's browser holds the rest); the output drive needs 5 GB, C: 2 GB.
set SOUP_RAM_MIN_MB=2048
if not exist "E:\Projects\TanitAD-artifacts\dagland\soup\logs" mkdir "E:\Projects\TanitAD-artifacts\dagland\soup\logs"
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/soup/soup_chain.sh %* < /dev/null > /e/Projects/TanitAD-artifacts/dagland/soup/logs/soup_chain.launch.out 2>&1" < NUL
