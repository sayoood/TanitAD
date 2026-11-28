#!/bin/bash
# End-of-task pod cleanup: stop the closed-loop services + renderer by explicit
# port->PID (NEVER pkill -f), verify GPU idle, release the gpu_lock. No deletions
# of the baseline collection or the fine-tune artifacts.
set -uo pipefail
echo "=== stopping services by port ==="
for p in 6789 6007 6006 6011; do
  pid=$(ss -ltnpH "sport = :$p" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  if [ -n "${pid:-}" ]; then kill "$pid" 2>/dev/null && echo "killed :$p pid=$pid"; fi
done
sleep 4
echo "=== residual compute procs ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
echo "=== GPU mem ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null
echo "=== release lock ==="
gpu_lock.sh release gate1-proto
gpu_lock.sh status
