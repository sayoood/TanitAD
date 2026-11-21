#!/bin/bash
# resume_after_refcv9.sh -- MM, 2026-11-21. The N1DM regression was stopped at 14:49Z so RefCV9's pinned suite (writing on F:) has room.
# Resume it once that suite's pytest (cmdline contains "junitxml=F:\refcv") has been absent for 3 checks 60 s apart AND F: has >= 6.5 GB free,
# then run run_regress_rest.sh (it re-runs every file without a summary line, incl. the stopped test_v8f_ego_perturb). Log: logs/resume.log.
B=F:/TanitAD-scratch/n1dc/build/n1dm_b1; LOG=$B/logs/resume.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
running() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*junitxml=F:\\refcv*' }).Count" | tr -d '\r '; }
say "start; waiting for the RefCV9 suite"
n=0
for i in $(seq 1 600); do
  c=$(running); f=$(df -m /f | tail -1 | awk '{print $4}')
  if [ "$c" = "0" ] && [ "$f" -ge 6500 ]; then n=$((n+1)); else n=0; fi
  [ $n -ge 3 ] && break
  sleep 60
done
say "suite gone for 3 checks (count $c, F: free ${f} MB); resuming the regression"
bash $B/run_regress_rest.sh >> $LOG 2>&1
say "run_regress_rest.sh ended rc=$?"
