@echo off
rem Detached runner of the r2 self-test + mutation proof (CPU only). Ends with the line ZZWRAPPER-END rc=<n> in logs\selftest_chain_r2.out (and ZZSELFTEST-R2-END PASS|FAIL from the script itself).
set CUDA_VISIBLE_DEVICES=-1
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" -u "E:\Projects\TanitAD-artifacts\dagorc_r2\gen\selftest_chain.py" > "E:\Projects\TanitAD-artifacts\dagorc_r2\logs\selftest_chain_r2.out" 2>&1
echo ZZWRAPPER-END rc=%ERRORLEVEL% >> "E:\Projects\TanitAD-artifacts\dagorc_r2\logs\selftest_chain_r2.out"
