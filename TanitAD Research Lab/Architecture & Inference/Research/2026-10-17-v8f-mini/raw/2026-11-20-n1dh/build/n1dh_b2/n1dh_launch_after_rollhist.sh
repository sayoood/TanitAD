#!/bin/bash
# MM 2026-11-21: start the N1DH chain only after the ROLLHIST chain has ended in DONE (PI go: "Yes, after lever C"). STAGED, NOT STARTED. Modelled on /root/rollhist/rh_launch_after_hpf1.sh (read, not edited).
# Done signal: the LAST line of /root/rollhist/rh_chain.log is `ZZ ROLLHIST DONE ZZ`. If the last line is `ZZ ROLLHIST FAIL ZZ ...` this gate does NOT start the chain (the PI's order is "after lever C"): it logs and exits 1, and the MM decides.
# Cap 40 h from start (exit 3, nothing started). Then it starts the marker watcher (which sets the marker) and the chain, exactly as the other launchers do. Read-only on /root/rollhist. Writes only n1dh_launch_after_rollhist.log.
L=/root/strat1/n1dh/n1dh_launch_after_rollhist.log
C=/root/rollhist/rh_chain.log
END=$(( $(date +%s) + 144000 ))
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
say "gate start: waiting for the last line of $C to be ZZ ROLLHIST DONE ZZ"
while :; do
  t=$(tail -n1 "$C" 2>/dev/null)
  case "$t" in
    "ZZ ROLLHIST DONE ZZ"*) say "ROLLHIST done: $t"; break;;
    "ZZ ROLLHIST FAIL ZZ"*) say "ROLLHIST FAILED, N1DH not started: $t"; exit 1;;
  esac
  if [ "$(date +%s)" -gt "$END" ]; then say "cap 40 h reached without ZZ ROLLHIST DONE ZZ; N1DH not started"; exit 3; fi
  sleep 30
done
nohup setsid bash /root/strat1/n1dh/n1dh_marker_watch.sh < /dev/null > /dev/null 2>&1 &
sleep 1
nohup setsid bash /root/strat1/n1dh/n1dh_chain.sh < /dev/null > /dev/null 2>&1 &
say "marker watcher and chain started"
