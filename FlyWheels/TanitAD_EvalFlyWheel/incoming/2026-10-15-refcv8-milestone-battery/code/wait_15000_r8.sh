#!/bin/sh
# Waits until Thor's trainer has FINISHED writing ckpt_15000.pt and ckpt_ema_15000.pt (both present, sizes unchanged over two polls
# 3 min apart -- torch.save there is not atomic, refc_v3_train.py:10834-10839), pulls them read-only (thor_pull_r8.sh: md5 on both
# sides), then starts queue_15000_r8.sh (EMA battery, then G0 / wrapper clause / raw battery).  Log: chain/wait_15000_r8.log
S=15000; KIT=/e/refcv8_eval_kit; P=/c/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code
RUN=/home/nvidia/refcv8_run/runs/refcv8-r101-s0; LOG=$KIT/chain/wait_15000_r8.log
mk() { echo "ZZW15$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mk START "pid $$"; prev=""; n=0
while :; do
  cur=$(timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=20 tanitad-thor "stat -c %s $RUN/ckpt_$S.pt $RUN/ckpt_ema_$S.pt 2>/dev/null | tr '\n' ' '")
  set -- $cur
  if [ $# -eq 2 ] && [ "$1" -gt 300000000 ] && [ "$2" -gt 300000000 ] && [ "$cur" = "$prev" ]; then mk STABLE "$cur"; break; fi
  prev="$cur"; n=$((n+1)); [ $((n % 10)) -eq 1 ] && mk POLL "sizes '$cur'"
  [ "$n" -ge 480 ] && { mk TIMEOUT "24 h without a stable pair"; exit 4; }
  sleep 180
done
sh "$P/thor_pull_r8.sh" $S > "$KIT/chain/wait15_pull.out" 2>&1
PR=$(tail -n 1 "$KIT/chain/wait15_pull.out")
case "$PR" in pull:ok:*) mk PULLOK "$PR" ;; *) mk PULLFAIL "$PR"; exit 5 ;; esac
[ -s "$KIT/ckpt/ckpt_ema_$S.pt" ] && [ -s "$KIT/ckpt/ckpt_$S.md5" ] || { mk PULLNOFILES "ema or md5 missing"; exit 5; }
mk QUEUE "starting queue_15000_r8.sh"
Q_NOWAIT=1 sh "$P/queue_15000_r8.sh" > "$KIT/chain/queue_15000_r8.out" 2>&1
mk END "queue returned"
