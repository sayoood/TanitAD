#!/usr/bin/env bash
# On-policy scorer pipeline SUPERVISOR, pod-local (PI decision 2026-09-26, B).
#   1 proposal dump (GPU, nice 19) + W0 rank-0 and W1 rank-1 labellers (CPU, nice 19, 1 thread each)
# Restarts a dead child every 60 s. ⛔ MEMORY GUARD: the container's cgroup limit is 50 GB and the
# kernel OOM killer would take the LARGEST process -- the live trainer. Above MEM_HIGH the supervisor
# stops the newest labeller and starts none until memory is back under MEM_LOW.
# Stop everything: touch $ROOT/STOP. Launch: setsid nohup bash op_pipeline.sh > $ROOT/logs/sup.out 2>&1 &
set -u
source /workspace/refe-op/code/op_env.sh
ROOT=/workspace/data/refe_navtrain/onpolicy
RUN=/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3
W0=${W0:-2}; W1=${W1:-2}
MEM_HIGH=${MEM_HIGH:-44000000000}; MEM_LOW=${MEM_LOW:-40000000000}
mkdir -p "$OP_QUEUE" "$OP_OUT" "$ROOT/logs" "$ROOT/pids"
say() { echo "$(date -u +%FT%TZ) $*" >> "$ROOT/logs/supervisor.log"; }
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
# children are started WITHOUT setsid: nohup and nice exec, so $! IS the python's pid
start_dump() {
  nohup nice -n 19 "$OP_PY" "$PKG/refe/onpolicy_dump.py" --run-dir "$RUN" --targets "$OP_BANK" \
    --images /workspace/data/navtrain_pixels --calib "$OP_BANK/calib_table.json" --queue "$OP_QUEUE" \
    --chunk 16 --max-pending 2 >> "$ROOT/logs/dump.log" 2>&1 < /dev/null &
  echo $! > "$ROOT/pids/dump.pid"; say "started dump pid $!"
}
start_lab() {
  nohup nice -n 19 "$OP_PY" "$PKG/refe/onpolicy_label.py" --queue "$OP_QUEUE" --out "$OP_OUT" \
    --rank "$1" --worker "$2" >> "$ROOT/logs/label_r$1_$2.log" 2>&1 < /dev/null &
  echo $! > "$ROOT/pids/label_r$1_$2.pid"; say "started label r$1 $2 pid $!"
}
say "supervisor up (pid $$): W0=$W0 W1=$W1 MEM_HIGH=$MEM_HIGH"
held=0
while [ ! -f "$ROOT/STOP" ]; do
  mem=$(cat /sys/fs/cgroup/memory.current)
  if [ "$mem" -gt "$MEM_HIGH" ]; then
    for p in $(ls -t "$ROOT"/pids/label_*.pid 2>/dev/null | head -1); do
      alive "$p" && { kill "$(cat "$p")"; say "MEMORY GUARD: $mem > $MEM_HIGH, stopped $(basename "$p")"; }
    done
    held=1
  elif [ "$held" = 1 ] && [ "$mem" -lt "$MEM_LOW" ]; then
    held=0; say "memory back to $mem: restarts allowed"
  fi
  alive "$ROOT/pids/dump.pid" || start_dump
  if [ "$held" = 0 ]; then
    for i in $(seq 0 $((W0 - 1))); do alive "$ROOT/pids/label_r0_w$i.pid" || start_lab 0 "w$i"; done
    for i in $(seq 0 $((W1 - 1))); do alive "$ROOT/pids/label_r1_w$i.pid" || start_lab 1 "w$i"; done
  fi
  sleep 60
done
say "STOP file seen: stopping children"
for p in "$ROOT"/pids/*.pid; do alive "$p" && kill "$(cat "$p")"; done
say "supervisor exit"
