#!/usr/bin/env bash
# Dev-box poll of the pod-side Stage B chain (HIERREP). Exits when STAGEB_DONE.txt or STAGEB_STOP.txt exists on the pod,
# or when at least $1 arms are done (default 99 = only on a marker). Prints the marker, the status line and the log tail.
# Reads the pod address only through the wrapper; prints no address. Usage: bash poll_stageB.sh [n_arms]
NEED=${1:-99}
P=/c/Users/Admin/v8pod/pssh.sh
FAILS=0
while true; do
  OUT=$(bash "$P" 'cd /workspace/v8proof/hierrep 2>/dev/null || exit 9; M=none; [ -f STAGEB_DONE.txt ] && M=DONE; [ -f STAGEB_STOP.txt ] && M=STOP; N=$(grep -o "\"n_done\":[0-9]*" STAGEB_STATUS.json 2>/dev/null | cut -d: -f2); echo "POLL marker=$M n_done=${N:-0}"' 2>/dev/null | sed '/hf_/d' | grep '^POLL ' | tail -n 1)
  if [ -z "$OUT" ]; then
    FAILS=$((FAILS + 1))
    if [ "$FAILS" -ge 12 ]; then echo "ZZ POLL-GAVE-UP no answer from the pod in 12 polls (2 h); the watcher exit is not evidence about the chain"; exit 3; fi
  else
    FAILS=0
    M=$(echo "$OUT" | sed 's/.*marker=\([A-Za-z]*\).*/\1/')
    N=$(echo "$OUT" | sed 's/.*n_done=\([0-9]*\).*/\1/')
    if [ "$M" != "none" ] || [ "${N:-0}" -ge "$NEED" ]; then
      echo "$OUT"
      bash "$P" 'cd /workspace/v8proof/hierrep; cat STAGEB_DONE.txt STAGEB_STOP.txt 2>/dev/null; cat STAGEB_STATUS.json 2>/dev/null; tail -n 12 stageB.log' 2>/dev/null | sed '/hf_/d' | cut -c1-300
      exit 0
    fi
  fi
  sleep 600
done
