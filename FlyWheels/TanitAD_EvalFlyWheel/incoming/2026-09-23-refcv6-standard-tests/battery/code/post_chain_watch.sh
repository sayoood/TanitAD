#!/bin/sh
# Unattended post-bank watcher (CPU, plus the A4 L3 GPU rolls behind the dev-box gate). It reads the
# chain's markers from raw/chain.log (it never writes there) and writes its own to raw/post_watch.log.
#   step5000  banked -> post_tag.sh step5000   (A3 recompute: its panels predate A3; A4 levers)
#   step30000 banked -> post_tag.sh step30000, then L3 for step5000 and step30000 (the chain is now
#                       waiting on Thor for the final, so the GPU is free)
#   final     banked -> post_tag.sh final, then L3 for the final
# Markers are opaque ZZ...ZZ tokens; the grep target (chain.log) never contains this script's text.
B=/c/Users/Admin/ev6_battery
LOG=$B/raw/chain.log
WL=$B/raw/post_watch.log
wait_for() { until grep -qE "$1" $LOG; do sleep 120; done; }
echo "ZZWATCHSTARTZZ $(date +%FT%T)" >> $WL
wait_for 'ZZBANKED_step5000ZZ|ZZBANKFAIL_step5000ZZ|ZZB30000DONEZZ|ZZCHAINMENDZZ'
grep -q 'ZZBANKED_step5000ZZ' $LOG && sh $B/code/post_tag.sh step5000
wait_for 'ZZBANKED_step30000ZZ|ZZBANKFAIL_step30000ZZ|ZZCHAINMENDZZ'
if grep -q 'ZZBANKED_step30000ZZ' $LOG; then
  sh $B/code/post_tag.sh step30000
  sh $B/code/l3_tag.sh step5000 D:/refcv6_eval_kit/ckpt/ckpt_5000.pt
  sh $B/code/l3_tag.sh step30000 D:/refcv6_eval_kit/ckpt/ckpt_30000.pt
fi
wait_for 'ZZBANKED_finalZZ|ZZBANKFAIL_finalZZ|ZZNOFINALZZ|ZZCHAINMENDZZ'
if grep -q 'ZZBANKED_finalZZ' $LOG; then
  sh $B/code/post_tag.sh final
  sh $B/code/l3_tag.sh final D:/refcv6_eval_kit/ckpt/ckpt_final.pt
fi
echo "ZZWATCHENDZZ $(date +%FT%T)" >> $WL
