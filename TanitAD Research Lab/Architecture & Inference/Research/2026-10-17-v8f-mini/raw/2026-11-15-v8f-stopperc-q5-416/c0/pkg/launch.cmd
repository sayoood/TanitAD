@echo off
set "S0P_SSH=C:\Program Files\Git\usr\bin\ssh.exe"
set "S0P_SFTP=C:\Program Files\Git\usr\bin\sftp.exe"
"C:\Program Files\Git\usr\bin\bash.exe" "C:\Users\Admin\q5c0\q5c0_chain.sh" < NUL > "C:\Users\Admin\q5c0\state\chain.console.log" 2>&1
