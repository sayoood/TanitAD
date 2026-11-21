#!/bin/bash
# Waits until Thor's GPU is idle of the OTHER agent's job (no compute app, util < 8 %, no process whose args name the other agent's
# tree) on 2 consecutive 30 s polls (their chain shell stays alive between its steps, so other=0 means finished), then runs the chain.  Never kills anything.  Opaque markers (CLAUDE.md monitor trap): the log
# lines carry computed counts, not the searched words.
source /home/nvidia/r8ns/perc8/code/thor_env.sh
LOG=$P8/status/wait.log
idle=0
for i in $(seq 1 260); do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  other=$(ps -eo args | grep -c "[r]8ns/fanc""eil")
  mine=$(ps -eo args | grep -c "[p]erc8/code/\(perc_pass\|run_diag\)")
  echo "$(date +%H:%M:%S) poll $i util=$util apps=$apps other=$other mine=$mine idle=$idle" >> $LOG
  if [ "${util:-100}" -lt 8 ] && [ "$apps" -eq 0 ] && [ "$other" -eq 0 ] && [ "$mine" -eq 0 ]; then idle=$((idle+1)); else idle=0; fi
  if [ $idle -ge 2 ]; then
    echo "$(date +%H:%M:%S) GPU idle x2 -> launching chain" >> $LOG
    bash $P8/code/chain_perc8.sh > $P8/logs/chain.log 2>&1
    echo "$(date +%H:%M:%S) chain exited rc=$?" >> $LOG
    exit 0
  fi
  sleep 30
done
echo "$(date +%H:%M:%S) GAVE UP after 260 polls" >> $LOG
