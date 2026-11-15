#!/usr/bin/env bash
# pod_tokprobe_wait.sh : runs ON THE POD. Waits (up to 14 h) for the marker of the DISTRULE queue (card ADD26), then starts the one TOKPROBE job
# (card ADD27 with note N1: main pass, control p3, then the sensitivity pass). Never more than three GPU jobs: it also waits while three compute
# processes are on the GPU. pod_tokprobe_run.sh carries the memory guard (20 GB free in the container) and the finished-marker guard.
M=/workspace/v8proof/distrule
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] tokprobe-wait: $*"; }
ok=0
for i in $(seq 1 1680); do
  if [ -f "$M/DISTRULE_DONE.txt" ] || [ -f "$M/DISTRULE_STOP.txt" ]; then ok=1; break; fi
  sleep 30
done
[ $ok = 1 ] || { say "no DISTRULE marker after 14 h; nothing started"; exit 1; }
say "DISTRULE marker present: $(ls $M | grep -E '^DISTRULE_(DONE|STOP)\.txt$' | tr '\n' ' ')"
for i in $(seq 1 240); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "${n:-0}" -lt 3 ] && break
  sleep 60
done
say "GPU compute processes: ${n:-0}; starting the TOKPROBE job"
cd /workspace/v8proof/probe || exit 1
for try in 1 2 3 4 5 6; do
  SENS=1 bash pod_tokprobe_run.sh
  rc=$?
  [ -f /workspace/v8proof/runs/tokprobe/TOKPROBE_DONE.txt ] && break
  # the one refusal that a wait can cure: container memory under the floor (another job is loading)
  grep -q 'free memory' /workspace/v8proof/runs/tokprobe/TOKPROBE_STOP.txt 2>/dev/null || break
  say "memory floor refusal (try $try); waiting 10 min"
  sleep 600
done
say "pod_tokprobe_run.sh ended rc $rc"
