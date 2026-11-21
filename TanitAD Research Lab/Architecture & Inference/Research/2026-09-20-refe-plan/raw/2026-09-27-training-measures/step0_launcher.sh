#!/usr/bin/env bash
# Step 0 window launcher: from START (local HHMM), when NO python process holds a CUDA context and utilisation <= 20 %
# (3 samples in a row, 10 s apart), run eval/fan_speed_swap_probe.py with --deadline DEADLINE. The probe itself
# refuses (rc 3) with < 4.5 GB free GPU memory; then wait and retry. Never STARTS at or after GIVEUP.
#   bash step0_launcher.sh <log> [START=1545] [GIVEUP=1612] [DEADLINE=16:20]
# 2026-09-27 15:45-16:12: NOT started -- 138 of 152 samples had another stream's python on the GPU (step0_window.log).
PKG="D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
LOG="$1"
START="${2:-1545}"
GIVEUP="${3:-1612}"
DEADLINE="${4:-16:20}"
say() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }
say "launcher up (pid $$); window start $START, give up $GIVEUP, probe deadline $DEADLINE"
while [ "$(date +%H%M)" -lt "$START" ]; do sleep 20; done
ok=0
while :; do
  if [ "$(date +%H%M)" -ge "$GIVEUP" ]; then
    say "not started by $GIVEUP -- probe NOT run in this window"
    echo "ZZLAUNCH_NOT_STARTED" >> "$LOG"
    exit 3
  fi
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -i python | tr -d '\r' | tr '\n' ';')
  q=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
  util=${q%%,*}
  mem=${q##*,}
  say "python_compute_apps=[${apps}] util=$util mem_used_MiB=$mem"
  if [ -z "$apps" ] && [ "${util:-100}" -le 20 ]; then ok=$((ok + 1)); else ok=0; fi
  if [ "$ok" -ge 3 ]; then
    say "GPU free -- starting the probe"
    (cd "$PKG/eval" && "$PY" fan_speed_swap_probe.py --device cuda --deadline "$DEADLINE" >> "$LOG" 2>&1)
    rc=$?
    say "PROBE_EXIT=$rc"
    if [ "$rc" -ne 3 ]; then
      echo "ZZLAUNCH_DONE rc=$rc" >> "$LOG"
      exit "$rc"
    fi
    say "probe refused to start (rc 3: GPU memory or deadline) -- waiting again"
    ok=0
  fi
  sleep 10
done
