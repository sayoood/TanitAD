#!/bin/bash
# reps2_marker_watch.sh -- the pod owner's priority marker for the REPS2 chain. FORK of /root/rollhist/rh_rl_marker_watch.sh (same protocol, read not edited). STAGED, NOT STARTED: the MM starts it BEFORE the chain launch.
# It sets /root/N1DW_READY so the other session's (TanitADRuled) NEXT slice skips (its current slice runs to its end), writes chains/reps2_marker_mine (so the chain may clear the marker itself on a refusal),
# then reads ONLY the chain-log lines written after it starts. When the chain logs "GPU queue lock held to the end of the chain" it removes the marker and exits 0. If the chain ends (a line "ZZ DSCALE10K REPS2 ...")
# before that line, or no blockreps2.sh process exists after the 120 s grace, it removes the marker and exits non-zero. The poll is every 10 s (a refusal hands the GPU back within seconds). Cap 30 h.
# It removes the marker ONLY if it set it itself (chains/reps2_marker_mine), never another session's. Writes only its own log. usage (MM):
#   nohup setsid bash /root/dscale10k/blockreps2/reps2_marker_watch.sh < /dev/null > /dev/null 2>&1 &
# Dry run: DS_ROOT / DS_MARKER / DS_CHAINLOG / DS_PATTERN / DS_GRACE / DS_POLL override (dryrun_reps2.sh).
D=${DS_ROOT:-/root/dscale10k}
M=${DS_MARKER:-/root/N1DW_READY}
CL=${DS_CHAINLOG:-$D/blockreps2.log}
LOG=${DS_WLOG:-$D/reps2_marker_watch.log}
MINE=$D/chains/reps2_marker_mine
PAT=${DS_PATTERN:-blockreps2[.]sh}
GRACE=${DS_GRACE:-120}
POLL=${DS_POLL:-10}
mkdir -p $D/chains
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
clear() { if [ -e $MINE ]; then rm -f $M $MINE; fi; }
touch $CL
B=$(wc -l < $CL)
touch $M $MINE; say "marker set ($M); chain-log baseline $B lines"
sleep $GRACE
for i in $(seq 1 ${DS_ITER:-12960}); do
  NEW=$(tail -n +$((B + 1)) $CL 2>/dev/null)
  if printf '%s\n' "$NEW" | grep -q "GPU queue lock held to the end of the chain"; then
    clear; say "the REPS2 chain holds the GPU lock; marker removed"; exit 0
  fi
  case "$(printf '%s\n' "$NEW" | tail -n1)" in
    "ZZ DSCALE10K REPS2 "*) clear; say "the REPS2 chain ended before the lock line; marker removed"; exit 1;;
  esac
  if ! grep -q "$PAT" /proc/[0-9]*/cmdline 2>/dev/null; then
    clear; say "no blockreps2.sh process; marker removed"; exit 2
  fi
  sleep $POLL
done
clear; say "30 h cap; marker removed"; exit 3
