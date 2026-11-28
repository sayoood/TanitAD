#!/bin/bash
# dev-box loop: exits when STAGEA_DONE.txt or STAGEA_STOP.txt exists on the pod; prints that marker and the last 5 log lines.
# Exit code 0 = DONE, 2 = STOP. Prints no address. Not started by the ops agent: the lead arms it.
P="bash /c/Users/Admin/v8pod/pssh.sh"
HR=/workspace/v8proof/hierrep
while :; do
  r=$($P "if [ -f $HR/STAGEA_DONE.txt ]; then echo DONE; elif [ -f $HR/STAGEA_STOP.txt ]; then echo STOP; else echo RUN; fi" </dev/null 2>/dev/null | tail -n 1)
  case "$r" in
    DONE) echo "== STAGEA_DONE.txt"; $P "cat $HR/STAGEA_DONE.txt" </dev/null | sed '/hf_/d'
          echo "== last 5 log lines (stageA.log)"; $P "tail -n 5 $HR/stageA.log" </dev/null | sed '/hf_/d'; exit 0;;
    STOP) echo "== STAGEA_STOP.txt"; $P "cat $HR/STAGEA_STOP.txt" </dev/null | sed '/hf_/d'
          echo "== last 5 log lines (stageA.log)"; $P "tail -n 5 $HR/stageA.log" </dev/null | sed '/hf_/d'; exit 2;;
  esac
  sleep 600
done
