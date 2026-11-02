@echo off
rem usage: run_group.cmd <tree-dir> <tag> <test files...>   (CPU only; writes logs\<tag>.log and an "exit N" marker line)
set "TREE=%~1"
set "TAG=%~2"
shift
shift
set CUDA_VISIBLE_DEVICES=-1
cd /d "%TREE%"
if not exist "F:\TanitAD-scratch\recover_ft\bt_%TAG%" mkdir "F:\TanitAD-scratch\recover_ft\bt_%TAG%"
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" -I -m pytest -q -rfE --durations=5 --basetemp="F:\TanitAD-scratch\recover_ft\bt_%TAG%" -p no:cacheprovider %1 %2 %3 %4 %5 %6 %7 %8 %9 > "F:\TanitAD-scratch\recover_ft\logs\%TAG%.log" 2>&1
echo exit %errorlevel% >> "F:\TanitAD-scratch\recover_ft\logs\%TAG%.log"
