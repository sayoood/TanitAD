#!/bin/bash
# One-line status of the S1 chain (refe/s1aux_pod.sh), from its chain.log and its lock.  Read-only: it starts nothing, stops nothing, touches nothing on the GPU.
#   bash s1aux_status.sh [work-dir]          (default /workspace/data/s1aux)
# Prints  ZZS1AUX_STATUS <state> arms_done=<...> now=<...> last=<the last marker line>   and exits:
#   0 running   (the chain holds its lock; last marker is not ZZS1AUX_WAITING)      9 waiting (the chain holds its lock and waits for RL-S1 / P2 / the GPU)
#   10 done     (the LAST chain start ended with ZZS1AUX_DONE)
#   11 refused  (the LAST chain start ended with ZZS1AUX_REFUSED: the line names the stage and why)
#   12 dead     (the last start has no terminal marker and nobody holds the lock: the process was killed; a re-run is safe)
#   13 none     (no chain.log)
# For a Monitor / until-loop: stop on exit codes 10, 11, 12.  The markers are anchored on the timestamp, so a command line that spells them cannot match.
R="${1:-/workspace/data/s1aux}"; L="$R/chain.log"
[ -f "$L" ] || { echo "ZZS1AUX_STATUS none no_chain_log"; exit 13; }
RX='^[0-9T:-]+ ZZS1AUX_(START|DONE|REFUSED)'
last_term="$(grep -nE "$RX" "$L" | tail -n 1)"
last_line="$(grep -E '^[0-9T:-]+ ZZS1AUX_' "$L" | tail -n 1 | cut -c1-260)"
done_arms="$(grep -E '^[0-9T:-]+ ZZS1AUX_ARM_DONE ' "$L" | sed 's/.*ZZS1AUX_ARM_DONE //' | awk '!s[$1]++ {printf "%s%s", (n++ ? "," : ""), $1}')"
now="$(grep -E '^[0-9T:-]+ ZZS1AUX_ARM_(GATE_OK|TRAIN_START|TRAINED|DONE|RESTART) ' "$L" | tail -n 1 | awk '{ sub(/^ZZS1AUX_ARM_/, "", $2); print $3 ":" $2 }')"
held=no
if [ -e "$R/chain.lock" ] && ! ( flock -n 9 ) 9<> "$R/chain.lock" 2> /dev/null; then held=yes; fi
case "$last_term" in
  *ZZS1AUX_DONE*)    state=done; rc=10 ;;
  *ZZS1AUX_REFUSED*) state=refused; rc=11 ;;
  *)
    if [ "$held" = yes ]; then
      if echo "$last_line" | grep -qE '^[0-9T:-]+ ZZS1AUX_WAITING'; then state=waiting; rc=9; else state=running; rc=0; fi
    else state=dead; rc=12; fi ;;
esac
echo "ZZS1AUX_STATUS $state arms_done=${done_arms:--} now=${now:--} stages_ok=$(ls "$R/stage"/*.ok 2> /dev/null | wc -l) last=$last_line"
exit $rc
