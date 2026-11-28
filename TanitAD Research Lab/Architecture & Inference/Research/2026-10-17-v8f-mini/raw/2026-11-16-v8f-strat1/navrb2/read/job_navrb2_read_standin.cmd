@echo off
rem dry run of the NAVRB2 read on the stand-in files (make_standin.py output): a code check, never a result
cd /d F:\TanitAD-scratch\strat1\navrb2\read
set NAVRB2_STANDIN_ROOT=F:/TanitAD-scratch/strat1/navrb2/read/standin
if not exist standin\out mkdir standin\out
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/strat1/navrb2/read/launch_navrb2_read.sh >> standin\out\navrb2_read_stdout.log 2>&1
echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> standin\out\navrb2_read_stdout.log
