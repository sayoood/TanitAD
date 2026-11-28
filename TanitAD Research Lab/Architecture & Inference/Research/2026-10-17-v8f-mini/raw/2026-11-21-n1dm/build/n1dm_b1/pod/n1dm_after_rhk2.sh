#!/bin/bash
# n1dm_after_rhk2.sh -- MM 2026-11-21 (replaces n1dm_after_rhk.sh once n1dm_chain_early.sh exists). PI go "Run both" + "Dont let the pod idle, use it effectively".
# "ZZ ROLLHIST-K ..." line (DONE or FAIL: N1DM runs either way), then runs n1dm_launch.sh (marker watcher + chain; the chain takes /root/gpu_queue.lock).
# Meanwhile it keeps /root/N1DW_READY set, so the other session's (TanitADRuled) NEXT slice skips between the two chains; it removes the marker only if it
# exits without launching (the N1DM watcher owns the marker after the launch). Writes only its own log. Cap 15 h.
M=/root/N1DW_READY
CL=/root/rollhist/rh_k_chain.log
LOG=/root/strat1/n1dm/after_rhk2.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
launched=0
trap 'if [ "$launched" != 1 ]; then rm -f $M; say "exiting without a launch; marker removed"; fi' EXIT
[ -e /root/strat1/n1dm/n1dm_go.token ] || { say "no n1dm_go.token; refusing"; exit 2; }
touch $M; say "marker set ($M); waiting for the ROLLHIST-K chain to end"
done_line=""
for i in $(seq 1 1800); do
  last=$(tail -n1 $CL 2>/dev/null)
  touch $M
  case "$last" in "ZZ ROLLHIST-K "*) done_line=$last; break;; esac
  if ! grep -q 'rh_k_chain[.]sh' /proc/[0-9]*/cmdline 2>/dev/null; then
    sleep 30; last=$(tail -n1 $CL 2>/dev/null)
    case "$last" in "ZZ ROLLHIST-K "*) done_line=$last; break;; esac
    say "no rh_k_chain.sh process and no ZZ ROLLHIST-K line (last: $last): not launching"; exit 3
  fi
  sleep 30
done
[ -n "$done_line" ] || { say "15 h cap reached: not launching"; exit 4; }
say "ROLLHIST-K ended: $done_line"
# wait for the early chain (n1dm_chain_early.sh) to end, so two N1DM chains never run at once; the marker stays set meanwhile (cap 3 h)
for j in $(seq 1 360); do grep -q "n1dm_chain_early[.]sh" /proc/[0-9]*/cmdline 2>/dev/null || break; touch $M; sleep 30; done
say "early chain: $(tail -n1 /root/strat1/n1dm/n1dm_chain_early.log 2>/dev/null)"
sleep 20
touch $M
bash /root/strat1/n1dm/n1dm_launch.sh; rc=$?
say "n1dm_launch.sh rc=$rc: $(tail -n1 /root/strat1/n1dm/n1dm_launch.log 2>/dev/null)"
[ $rc = 0 ] && launched=1
exit $rc
