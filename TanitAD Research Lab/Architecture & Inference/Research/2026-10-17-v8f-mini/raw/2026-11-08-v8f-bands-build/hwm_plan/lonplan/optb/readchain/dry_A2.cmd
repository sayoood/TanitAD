@echo off
cd /d F:\TanitAD-scratch\actrsp\optb\readchain
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/actrsp/optb/readchain/dry_run.sh A >> F:\TanitAD-scratch\actrsp\optb\readchain\tmp\dry_A2.out 2>&1
echo rc=%ERRORLEVEL%>> F:\TanitAD-scratch\actrsp\optb\readchain\tmp\dry_A2.out
