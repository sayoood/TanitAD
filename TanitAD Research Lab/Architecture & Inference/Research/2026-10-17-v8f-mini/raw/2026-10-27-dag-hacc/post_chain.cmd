@echo off
rem Detached launcher for post/post_chain.sh (start it through WMI Win32_Process.Create with CreateFlags 0x08000200).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/daghacc/post/post_chain.sh < /dev/null" < NUL > E:\Projects\TanitAD-artifacts\daghacc\logs\post_chain.launch.out 2>&1
