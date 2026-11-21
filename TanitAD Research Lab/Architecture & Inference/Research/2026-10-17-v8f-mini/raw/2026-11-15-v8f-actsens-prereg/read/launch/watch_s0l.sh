#!/usr/bin/env bash
# MM watch: S0-L dev-box jobs end (S0L_L.done = trained-path L dumps + STOPDEC on both tags; S0L_LFOLD.done = folded route beside), or 90 min pass.
S=/e/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0l
for i in $(seq 1 90); do
  if [ -f "$S/S0L_L.done" ] && [ -f "$S/S0L_LFOLD.done" ]; then echo "S0L JOBS DONE $(date '+%F %T') :: L: $(cat $S/S0L_L.done) :: LFOLD: $(cat $S/S0L_LFOLD.done)"; exit 0; fi
  sleep 60
done
echo "S0L WATCH-TIMEOUT $(date '+%F %T') :: L: $(cat $S/S0L_L.done 2>/dev/null) :: LFOLD: $(cat $S/S0L_LFOLD.done 2>/dev/null)"
