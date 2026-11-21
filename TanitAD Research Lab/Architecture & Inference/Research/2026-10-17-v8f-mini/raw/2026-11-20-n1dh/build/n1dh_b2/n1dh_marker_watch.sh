#!/bin/bash
# n1dh_marker_watch.sh -- the pod owner's priority marker for the N1DH chain. Copied from /root/strat1/n1dc/n1dc_marker_watch2.sh (read, not edited). STAGED, NOT STARTED: n1dh_launch_after_rollhist.sh starts it, then the chain.
# It sets /root/N1DW_READY so the other session's (TanitADRuled) NEXT slice skips (its current slice runs to its end), then reads ONLY the chain-log lines written after it starts: when the chain logs
# "GPU queue lock held to the end of the chain" it removes the marker and exits 0; if the chain ends before that line, or no n1dh_chain.sh process exists, it removes the marker and exits non-zero. Cap 30 h. Writes only its own log.
M=/root/N1DW_READY
CL=/root/strat1/n1dh/n1dh_chain.log
LOG=/root/strat1/n1dh/marker_watch.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
touch $CL
B=$(wc -l < $CL)
touch $M; say "marker set ($M); chain-log baseline $B lines"
sleep 120
for i in $(seq 1 3600); do
  NEW=$(tail -n +$((B + 1)) $CL 2>/dev/null)
  if printf '%s\n' "$NEW" | grep -q "GPU queue lock held to the end of the chain"; then
    rm -f $M; say "the N1DH chain holds the GPU lock; marker removed"; exit 0
  fi
  case "$(printf '%s\n' "$NEW" | tail -n1)" in
    "ZZ N1DH "*) rm -f $M; say "the N1DH chain ended before the lock line; marker removed"; exit 1;;
  esac
  if ! grep -q 'n1dh_chain[.]sh' /proc/[0-9]*/cmdline 2>/dev/null; then
    rm -f $M; say "no n1dh_chain.sh process; marker removed"; exit 2
  fi
  sleep 30
done
rm -f $M; say "30 h cap; marker removed"; exit 3
