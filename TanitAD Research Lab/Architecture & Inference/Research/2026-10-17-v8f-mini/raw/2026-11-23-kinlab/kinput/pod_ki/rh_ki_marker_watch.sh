#!/bin/bash
# rh_ki_marker_watch.sh -- the pod owner's priority marker for the K-INPUT chain. In RA's form (/root/rollhist/rh_ra_marker_watch.sh), NOT RK's ADD123 take-over. STAGED, NOT STARTED: the MM starts it (launch_ki.sh) BEFORE the chain.
# What it does:
#   1. AT ITS START it sets /root/N1DW_READY ONLY IF ABSENT (noclobber create, atomic), with its own token as the content, and records the token in its own file ki_marker_mine. If the marker is PRESENT AND NOT ITS OWN at its start,
#      it sets nothing, logs that, and exits 0 (it neither removes nor overwrites a marker it did not set).
#   2. It reads ONLY the chain-log lines written after it started. When the chain logs "GPU queue lock held to the end of the chain" it clears the marker, ONLY IF the marker's content equals its token, and exits 0.
#   3. If the chain ends before that line (a last line `ZZ ROLLHIST-KI ...`), or no rh_ki_chain.sh process exists (after a settle time), it clears its own token (same rule) and exits NON-ZERO (1 = chain ended, 2 = no chain process).
#   4. Cap 30 h: it clears its own token and exits 4.
# Writes only its own log and token file (and the marker it created). usage (MM, via launch_ki.sh):  nohup setsid bash /root/rollhist/kit_ki/rh_ki_marker_watch.sh < /dev/null > /dev/null 2>&1 &
# The KIW_* overrides exist ONLY for the dev-box test (test_ki_marker_watch.sh, scratch paths); the launch sets none, so the defaults below are the pod's.
M=${KIW_MARKER:-/root/N1DW_READY}
MINE=${KIW_MINE:-/root/rollhist/ki_marker_mine}
CL=${KIW_CHAINLOG:-/root/rollhist/rh_ki_chain.log}
LOG=${KIW_LOG:-/root/rollhist/ki_marker_watch.log}
SETTLE=${KIW_SETTLE:-20}
POLL=${KIW_POLL:-30}
TOKEN="KI-$$-$(date +%s)"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
clear_mine() { if [ -e "$MINE" ] && [ -e "$M" ] && [ "$(cat "$M" 2>/dev/null)" = "$(cat "$MINE")" ]; then rm -f "$M"; say "marker cleared (own token)"; else say "marker not cleared: absent or not ours"; fi; rm -f "$MINE"; }
touch $CL
B=$(wc -l < $CL)
if ( set -C; printf '%s\n' "$TOKEN" > $M ) 2>/dev/null; then
  printf '%s\n' "$TOKEN" > $MINE
  say "watcher start (token $TOKEN); marker set ($M) with own token; chain-log baseline $B lines"
else
  say "watcher start (token $TOKEN); marker $M is present and not ours: nothing set, nothing to clear; exit 0"
  exit 0
fi
sleep $SETTLE
for i in $(seq 1 3600); do
  NEW=$(tail -n +$((B + 1)) $CL 2>/dev/null)
  if printf '%s\n' "$NEW" | grep -q "GPU queue lock held to the end of the chain"; then
    clear_mine; say "the ROLLHIST-KI chain holds the GPU lock"; exit 0
  fi
  case "$(printf '%s\n' "$NEW" | tail -n1)" in
    "ZZ ROLLHIST-KI "*) clear_mine; say "the ROLLHIST-KI chain ended before the lock line"; exit 1;;
  esac
  if ! grep -q 'rh_ki_chain[.]sh' /proc/[0-9]*/cmdline 2>/dev/null; then
    clear_mine; say "no rh_ki_chain.sh process"; exit 2
  fi
  sleep $POLL
done
clear_mine; say "30 h cap"; exit 4
