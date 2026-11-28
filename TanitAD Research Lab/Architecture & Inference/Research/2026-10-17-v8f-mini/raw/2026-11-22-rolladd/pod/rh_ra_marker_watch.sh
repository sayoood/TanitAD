#!/bin/bash
# rh_marker_watch.sh -- the pod owner's priority marker for the ROLLHIST-RA chain. Modelled on /root/strat1/n1dc/n1dc_marker_watch2.sh (read, not edited). STAGED, NOT STARTED: the MM starts it BEFORE the chain launch.
# It sets /root/N1DW_READY so the other session's (TanitADRuled) NEXT slice skips (its current slice runs to its end), then reads ONLY the chain-log lines written after it starts: when the chain logs
# "GPU queue lock held to the end of the chain" it removes the marker and exits 0; if the chain ends before that line, or no rh_ra_chain.sh process exists, it removes the marker and exits non-zero. Cap 30 h.
# Writes only its own log. usage (MM):  nohup setsid bash /root/rollhist/rh_marker_watch.sh < /dev/null > /dev/null 2>&1 &
M=/root/N1DW_READY
CL=/root/rollhist/rh_ra_chain.log
LOG=/root/rollhist/ra_marker_watch.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
touch $CL
B=$(wc -l < $CL)
touch $M; say "marker set ($M); chain-log baseline $B lines"
sleep 120
for i in $(seq 1 3600); do
  NEW=$(tail -n +$((B + 1)) $CL 2>/dev/null)
  if printf '%s\n' "$NEW" | grep -q "GPU queue lock held to the end of the chain"; then
    rm -f $M; say "the ROLLHIST-RA chain holds the GPU lock; marker removed"; exit 0
  fi
  case "$(printf '%s\n' "$NEW" | tail -n1)" in
    "ZZ ROLLHIST-RA "*) rm -f $M; say "the ROLLHIST-RA chain ended before the lock line; marker removed"; exit 1;;
  esac
  if ! grep -q 'rh_ra_chain[.]sh' /proc/[0-9]*/cmdline 2>/dev/null; then
    rm -f $M; say "no rh_ra_chain.sh process; marker removed"; exit 2
  fi
  sleep 30
done
rm -f $M; say "30 h cap; marker removed"; exit 3
