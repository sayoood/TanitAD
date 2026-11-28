@echo off
rem Detached launcher for pull_speedread.sh (start through WMI Win32_Process.Create). It waits for box 2 to be idle, pulls, then starts run_speedread.sh.
if not exist "F:\TanitAD-scratch\speedread" mkdir "F:\TanitAD-scratch\speedread"
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/speedread_run/pull_speedread.sh < /dev/null" < NUL > F:\TanitAD-scratch\speedread\pull.launch.out 2>&1
