#!/bin/bash
# rl_gsuite_locked.sh -- MM 2026-11-22. Runs rl_gsuite.sh (the fresh G-SUITE at cf27dbb7, CPU only) while holding /root/gpu_queue.lock, then releases it on exit,
# so the suite never overlaps a training job and the dry mint (DRY_MINT_PLAN.md B, which needs gsuite_result_rl.json) can run before launch_rl.sh.
# The chain's gsuite step keeps a passed result (gs_passed). Log: /root/rollhist/logs/rl_gsuite_locked.log
R=/root/rollhist
mkdir -p $R/logs
exec >> $R/logs/rl_gsuite_locked.log 2>&1
echo "$(date -u +%FT%TZ) waiting for the GPU queue lock"
exec 9>/root/gpu_queue.lock
flock 9 || { echo "ZZ RL GSUITE-LOCKED FAIL ZZ flock"; exit 1; }
echo "$(date -u +%FT%TZ) lock held; rl_gsuite.sh"
bash $R/rl_gsuite.sh; rc=$?
echo "$(date -u +%FT%TZ) rl_gsuite.sh rc=$rc; last line: $(tail -n 1 $R/logs/rl_gsuite.log 2>/dev/null)"
[ $rc -eq 0 ] && tail -n 1 $R/logs/rl_gsuite.log | grep -q "ZZ RL GSUITE DONE ZZ" && echo "ZZ RL GSUITE-LOCKED DONE ZZ" || echo "ZZ RL GSUITE-LOCKED FAIL ZZ rc=$rc"
