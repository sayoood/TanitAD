#!/bin/bash
# Standing P6 watcher for the PROXY-ONLY pod (2026-10-22). Zero model turns while healthy. Exits on the FIRST of:
#   POD_UNREACHABLE (3 failed probes ~15 min), TRAIN_STALE (train.log untouched > 20 min with no chain end-marker),
#   CHAIN_END (ZZP6_DONE / TRAIN_DONE / FAILED / FINAL_FAILED / REFUSED after $SINCE), or MAXIT polls (24 h) -> HEARTBEAT with the step.
SINCE="${1:-2026-10-22T23:00:00}"; MAXIT="${2:-288}"; P=C:/Users/Admin/par18/p6stage/work; fails=0
for i in $(seq 1 "$MAXIT"); do
  out=$(printf '%s\n' 'P=/workspace/data/refe_p6; now=$(date +%s); t=$(stat -c %Y $P/train.log 2>/dev/null || echo 0); e=$(awk -v s="'"$SINCE"'" '"'"'$1>s && (/ZZP6_(DONE|TRAIN_DONE|FAILED|FINAL_FAILED|REFUSED)/ || /REFUSED:/)'"'"' $P/chain.log | tail -1 | cut -c1-120 | tr " " "_"); st=$(grep -E "^ +step +[0-9]+" $P/train.log | tail -1 | awk "{print \$2}"); echo "ZZW age=$((now-t)) step=${st:-0} up=$(cut -d. -f1 /proc/uptime) END=${e}"' | bash $P/pp.sh 15 | grep '^ZZW ' | sed '/hf_[A-Za-z0-9]\{20\}/d' | tail -1)
  if [ -z "$out" ]; then fails=$((fails+1)); [ $fails -ge 3 ] && { echo "ZZP6W_POD_UNREACHABLE $(date -u +%FT%TZ)"; exit 2; }; sleep 300; continue; fi
  fails=0
  echo "$out" | grep -qE "END=[^ ]+" && { echo "ZZP6W_CHAIN_END $(date -u +%FT%TZ) :: $out"; exit 0; }
  age=$(echo "$out" | sed -E 's/.*age=([0-9-]+).*/\1/'); [ "$age" -gt 1200 ] && { echo "ZZP6W_TRAIN_STALE $(date -u +%FT%TZ) :: $out"; exit 1; }
  last="$out"; sleep 300
done
echo "ZZP6W_HEARTBEAT $(date -u +%FT%TZ) :: $last"
