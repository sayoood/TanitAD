@echo off
rem LEVELPROBE2 chain launcher via WMI Win32_Process.Create.  RESUME = launch again.
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagland/levelprobe/levelprobe2_chain.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/levelprobe/logs/chain2.launch.out 2>&1" < NUL
