#!/bin/bash
# rh_rk_marker_watch.sh -- the pod owner's priority marker for the KINLAB-RK chain. Modelled on rh_ra_marker_watch.sh (itself on /root/strat1/n1dc/n1dc_marker_watch2.sh). STAGED, NOT STARTED: the MM starts it (launch_rk.sh) BEFORE the chain.
# Differences from the RA watcher, and only these (the ADD123 chain, /root/dscale_lead/mm_add123_chain.sh, holds /root/N1DW_READY until its own end, with an EMPTY marker file and its own token file add123_marker_mine):
#   1. PHASE 1: it sets NOTHING while the ADD123 chain runs. It polls (every 2 s) the ADD123 log for the end: its last line `ZZ ADD123 ` (DONE | STOP | FAIL), or the earlier line `marker removed (mine)` which that chain writes right after it removes the marker (its tar
#      step runs between the two, so waiting for the last line alone would leave the marker absent for the tar's duration). Cap 13 h (the chain's wait is 12 h).
#   2. PHASE 2 (the take-over, no gap): in the SAME check that sees the end, it sets the marker ONLY IF IT IS ABSENT (noclobber create, atomic), with its own token as content, and records the token in its own file (rk_marker_mine).
#      If the marker is still present (the ADD123 chain's removal has not happened yet) it retries every 2 s for up to 10 min; it never overwrites and never removes a marker it did not set.
#   3. PHASE 3: as the RA watcher: it reads ONLY the chain-log lines written after it started; when the chain logs "GPU queue lock held to the end of the chain" it clears the marker and exits 0; if the chain ends before that line, or no rh_rk_chain.sh process
#      exists, it clears the marker and exits non-zero. Cap 30 h from the end of phase 1. THE G-SUITE (CPU, ~36 min) therefore runs under the marker, before the lock.
#   Clearing = only its OWN token: the marker is removed only if its content equals the content of rk_marker_mine.
# Writes only its own log and token file. usage (MM):  nohup setsid bash /root/rollhist/rh_rk_marker_watch.sh < /dev/null > /dev/null 2>&1 &
# The RKW_* overrides exist ONLY for the dev-box test (test_rk_marker_watch.sh, scratch paths); the launch (launch_rk.sh) sets none, so the defaults below are the pod's.
M=${RKW_MARKER:-/root/N1DW_READY}
MINE=${RKW_MINE:-/root/rollhist/rk_marker_mine}
ADD=${RKW_ADDLOG:-/root/dscale_lead/mm_add123_chain.log}
CL=${RKW_CHAINLOG:-/root/rollhist/rh_rk_chain.log}
LOG=${RKW_LOG:-/root/rollhist/rk_marker_watch.log}
SETTLE=${RKW_SETTLE:-20}
TOKEN="RK-$$-$(date +%s)"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
clear_mine() { if [ -e "$MINE" ] && [ -e "$M" ] && [ "$(cat "$M" 2>/dev/null)" = "$(cat "$MINE")" ]; then rm -f "$M"; say "marker cleared (own token)"; else say "marker not cleared: absent or not ours"; fi; rm -f "$MINE"; }
touch $CL
B=$(wc -l < $CL)
say "watcher start (token $TOKEN); chain-log baseline $B lines; waiting for the ADD123 chain's end in $ADD"
ended=0
for i in $(seq 1 23400); do
  if grep -q -e "ZZ ADD123 " -e "marker removed (mine)" $ADD 2>/dev/null; then
    # PHASE 2, in the same check: set the marker if absent (atomic noclobber create)
    for j in $(seq 1 300); do
      if ( set -C; printf '%s\n' "$TOKEN" > $M ) 2>/dev/null; then printf '%s\n' "$TOKEN" > $MINE; say "ADD123 ended; marker set ($M) with own token"; ended=1; break; fi
      sleep 2
    done
    [ $ended = 1 ] || { say "ADD123 ended but the marker stayed present for 10 min (not ours): not set"; ended=2; }
    break
  fi
  sleep 2
done
[ $ended = 0 ] && { say "13 h cap: the ADD123 chain did not end; nothing set"; exit 3; }
sleep $SETTLE
for i in $(seq 1 3600); do
  NEW=$(tail -n +$((B + 1)) $CL 2>/dev/null)
  if printf '%s\n' "$NEW" | grep -q "GPU queue lock held to the end of the chain"; then
    clear_mine; say "the ROLLHIST-RK chain holds the GPU lock"; exit 0
  fi
  case "$(printf '%s\n' "$NEW" | tail -n1)" in
    "ZZ ROLLHIST-RK "*) clear_mine; say "the ROLLHIST-RK chain ended before the lock line"; exit 1;;
  esac
  if ! grep -q 'rh_rk_chain[.]sh' /proc/[0-9]*/cmdline 2>/dev/null; then
    clear_mine; say "no rh_rk_chain.sh process"; exit 2
  fi
  sleep 30
done
clear_mine; say "30 h cap"; exit 4
