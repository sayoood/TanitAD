@echo off
rem OPTB3 read chain (launch through WMI, CreateFlags 0x08000200, stdin NUL). Sentinel: out\ZZ_OPTB3READ.txt (one line: ZZ OPTB3READ DONE ZZ or ZZ OPTB3READ FAIL ZZ <reason>), also the last line of optb3_driver.out.
cd /d F:\TanitAD-scratch\optb3_read
set PATH=C:\Program Files\Git\usr\bin;%PATH%
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/optb3_read/optb3_readchain.sh >> F:\TanitAD-scratch\optb3_read\optb3_driver.out 2>&1
