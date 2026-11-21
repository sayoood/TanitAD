@echo off
"C:\Program Files\Git\bin\bash.exe" -lc "bash /c/Users/Admin/hmc/hmc_chain.sh %1 < /dev/null" < NUL > C:\Users\Admin\hmc\logs\hmc_chain.launch.out 2>&1
