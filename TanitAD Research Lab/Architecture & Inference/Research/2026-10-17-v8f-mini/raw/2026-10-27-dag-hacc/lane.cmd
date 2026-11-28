@echo off
rem Detached launcher for rollout_lane.sh <seed> (start it through WMI Win32_Process.Create with CreateFlags 0x08000200).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/daghacc/rollout_lane.sh %1 < /dev/null" < NUL > E:\Projects\TanitAD-artifacts\daghacc\logs\lane%1.launch.out 2>&1
