@echo off
rem C0C-RE read waiter launcher (Master Mind): start through WMI Win32_Process.Create so it survives the caller.
"C:\Program Files\Git\bin\bash.exe" -c "bash /e/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/c0cre_read_waiter.sh < /dev/null > /e/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE/logs/c0cre_read_waiter.out 2>&1" < NUL
