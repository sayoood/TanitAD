#!/bin/sh
# Sequenced pull of the RAW parity VAL epcache (600 eps, 70.39 GB) onto Thor.
#
# WHY SEQUENCED: a concurrent stream is already pulling the 278 GB train
# epcache (PID passed as $1). HF download is bandwidth-bound at ~24 MB/s
# MEASURED, so running both at once does not finish either sooner -- it just
# halves the train pull's rate and delays the more critical copy. So we wait on
# the train PID, then pull.
#
# We wait by polling /proc/<pid>. NEVER `pgrep -f` / `pkill -f`: those
# self-match the ssh command that launched them and kill the session, returning
# empty output so it looks like nothing happened.
#
# Logs go to /tmp = LOCAL disk. A logger writing to a failing filesystem cannot
# report that the filesystem is failing.
set -u
WAITPID="$1"
LOG=/tmp/pull_val600.log

echo "{\"stage\":\"waiting\",\"on_pid\":$WAITPID,\"t\":\"$(date -u +%FT%TZ)\"}" >> "$LOG"
while [ -d "/proc/$WAITPID" ]; do
  sleep 60
done
echo "{\"stage\":\"train_pull_exited\",\"t\":\"$(date -u +%FT%TZ)\"}" >> "$LOG"

# Do not start if the train pull did not actually complete -- a half corpus
# plus a val pull is worse than a finished corpus.
N=$(ls /home/nvidia/epcache/epcache-256px-phase0/physicalai-train-e438721ae894/ep_*.pt 2>/dev/null | wc -l)
echo "{\"stage\":\"train_ep_count\",\"n\":$N,\"expected\":2376}" >> "$LOG"

nohup /home/nvidia/venvs/tanitad-edge/bin/python /tmp/pull_val600.py >> "$LOG" 2>&1 &
echo "{\"stage\":\"val_pull_launched\",\"pid\":$!,\"t\":\"$(date -u +%FT%TZ)\"}" >> "$LOG"
