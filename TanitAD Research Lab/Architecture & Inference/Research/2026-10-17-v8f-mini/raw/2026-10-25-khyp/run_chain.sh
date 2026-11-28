#!/bin/bash
# runs the five Phase-B stages strictly one after another (one python process at a time); stops at the first non-zero rc (incl. guard refusal rc 3)
O="C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/khyp"
for st in ${KHYP_STAGES:-eval0 train0 eval1 train1 read}; do
  bash "$O/run_phaseB.sh" "$st" < /dev/null >> "$O/logs/chain.log" 2>&1
  rc=$?
  tail -n 3 "$O/logs/chain.log" > /dev/null
  if [ "$rc" -ne 0 ]; then echo "CHAIN STOP at $st rc=$rc $(date '+%H:%M:%S')" >> "$O/logs/chain.log"; echo "CHAIN STOP at $st rc=$rc"; exit $rc; fi
done
echo "CHAIN DONE $(date '+%H:%M:%S')" >> "$O/logs/chain.log"; echo "CHAIN DONE"
