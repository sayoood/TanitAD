@echo off
rem Detached launcher (WMI Win32_Process.Create): waits for the SPEEDREAD pull to end and release BOX2_BUSY, then runs chain_v8f_dagoaccorc.sh (PI go 2026-10-27, DAG-ORC re-based on DAG-OACC by Amendment 1).
"C:\Program Files\Git\bin\bash.exe" -lc "bash /c/Users/Admin/box2_ship/run_dagoaccorc_after_speedread.sh < /dev/null" < NUL > C:\Users\Admin\box2_ship\logs\dagoaccorc_waiter.launch.out 2>&1
