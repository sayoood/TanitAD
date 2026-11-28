@echo off
rem E3 relaunch v3 2026-11-11 ~22:52Z: v2 (runA2) died at 33/261 on "forrtl error (200): window-CLOSE event" (WMI launch without
rem CreateFlags). Launched now with CREATE_NEW_PROCESS_GROUP|CREATE_NO_WINDOW. usr\bin\bash.exe direct (the Git\bin wrapper ^C trap).
set PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\mingw64\bin;%PATH%
if not exist F:\tanitad_scratch\tf_e3\runA3 mkdir F:\tanitad_scratch\tf_e3\runA3
if not exist F:\tanitad_scratch\tf_e3\runB3 mkdir F:\tanitad_scratch\tf_e3\runB3
"C:\Program Files\Git\usr\bin\bash.exe" F:/tanitad_scratch/tf_e3/t10/run.sh F:/tanitad_scratch/tf_e3/runA3/out driverl_test14_random test 0 3 > F:\tanitad_scratch\tf_e3\runA3\run.log 2>&1 < NUL
echo DONE_RC=%ERRORLEVEL% > F:\tanitad_scratch\tf_e3\runA3\DONE.txt
"C:\Program Files\Git\usr\bin\bash.exe" F:/tanitad_scratch/tf_e3/t10/run.sh F:/tanitad_scratch/tf_e3/runB3/out driverl_test14_random test 0 3 > F:\tanitad_scratch\tf_e3\runB3\run.log 2>&1 < NUL
echo DONE_RC=%ERRORLEVEL% > F:\tanitad_scratch\tf_e3\runB3\DONE.txt
