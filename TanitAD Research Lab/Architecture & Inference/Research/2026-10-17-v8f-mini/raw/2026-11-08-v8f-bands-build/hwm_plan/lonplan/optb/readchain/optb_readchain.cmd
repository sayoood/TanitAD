@echo off
rem OPTB read chain (MM launches through WMI, CreateFlags 0x08000200, stdin NUL). Sentinels: out\ZZ_STOP.txt / ZZ_DONE.txt / ZZ_ENDED_BY_BOX2_STOP.txt (own files); chain.log carries none.
cd /d F:\TanitAD-scratch\actrsp\optb\readchain
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/actrsp/optb/readchain/optb_readchain.sh >> F:\TanitAD-scratch\actrsp\optb\readchain\optb_readchain.out 2>&1
