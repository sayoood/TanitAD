@echo off
cd /d F:\TanitAD-scratch\actrsp\optb\readchain
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/actrsp/optb/readchain/dry_run.sh A >> F:\TanitAD-scratch\actrsp\optb\readchain\tmp\dry_all.out 2>&1
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/actrsp/optb/readchain/time_plan.sh >> F:\TanitAD-scratch\actrsp\optb\readchain\tmp\dry_all.out 2>&1
