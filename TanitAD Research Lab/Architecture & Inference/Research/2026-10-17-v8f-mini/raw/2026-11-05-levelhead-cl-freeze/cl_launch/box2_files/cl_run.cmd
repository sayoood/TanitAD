@echo off
rem LEVELHEAD arm CL: ONE trainer invocation (gate = the argv's --selfcheck-only twin, or the training run; the W probe is a gate-mode call) from its PINNED line. Called as: cmd /c D:\tanitad\cl\cl_run.cmd <gate|train> <tag> under %RUNSAFE%.
rem The exit code is the trainer's: the caller (cl_check.py gate / post) judges it AND the artifacts. Resumes from ckpt_last.pt when the same line is run again (the trainer's own rule).
setlocal enabledelayedexpansion
call D:\tanitad\env.cmd
set "MODE=%~1"
set "TAG=%~2"
set "RUNSM=D:\tanitad\runs\v8f_mini_full"
set "TRAIN=D:\tanitad\repo\stack\scripts\v8f_mini_train.py"
set "PINF=%RUNSM%\cl\pin\%TAG%.%MODE%.txt"
set "LOG=%RUNSM%\cl\logs\%TAG%.%MODE%.log"
set "X="
for /f "usebackq delims=" %%L in ("%PINF%") do set "X=%%L"
if not defined X (echo REFUSED: no pinned line at %PINF% & exit /b 2)
echo ===== RUN-START %MODE% %TAG% %DATE% %TIME% >> "%LOG%"
"%PY%" "%TRAIN%" !X! >> "%LOG%" 2>&1
set "RC=!ERRORLEVEL!"
echo ===== RUN-END %MODE% %TAG% rc=!RC! %DATE% %TIME% >> "%LOG%"
exit /b !RC!
