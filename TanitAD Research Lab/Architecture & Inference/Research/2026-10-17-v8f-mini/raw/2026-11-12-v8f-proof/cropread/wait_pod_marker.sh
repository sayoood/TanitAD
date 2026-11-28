#!/bin/bash
# wait_pod_marker.sh <pod dir> <done name> <stop name> [max rounds] -- lead's wait: one pssh call per round, the waiting is a remote loop (at most 500 s per call).
# Prints one line when a marker exists or the rounds are used up. Reads only; opens no result value.
set -u
DIR=$1; DONE=$2; STOP=$3; MAX=${4:-14}
PS=/c/Users/Admin/v8pod/pssh.sh
for r in $(seq 1 "$MAX"); do
  out=$(timeout 560 bash "$PS" "for i in \$(seq 1 25); do if [ -e $DIR/$DONE ]; then echo ZZ-MARK DONE; exit 0; fi; if [ -e $DIR/$STOP ]; then echo ZZ-MARK STOP; exit 0; fi; sleep 20; done; echo ZZ-MARK WAIT" 2>/dev/null | sed '/hf_/d' | tr -d '\r' | grep '^ZZ-MARK' | tail -n 1)
  case "$out" in
    "ZZ-MARK DONE") echo "ZZ WAIT-RESULT DONE round $r $(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit 0 ;;
    "ZZ-MARK STOP") echo "ZZ WAIT-RESULT STOP round $r $(date -u +%Y-%m-%dT%H:%M:%SZ)"; exit 3 ;;
  esac
done
echo "ZZ WAIT-RESULT TIMEOUT after $MAX rounds $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 4
