#!/usr/bin/env bash
# MM 2026-11-13 ~05:50Z: replaces sequencer3 (pid 248818, stopped by the MM) for the rest of A5HB.
# A5HC is NOT launched: at HB-s0's measured cost (3.24 s/update, ~10 L40-h) HC-s0 + HC-SHUF go over the PI's 16.7 L40-h cap; A5HC waits for the PI.
OUT=/workspace/data/refe_vdac; SQ=$OUT/a5_sequencer3.log
say() { echo "$(date -u +%FT%TZ) $*" >> $SQ; }
W=248857
[ "$(ps -o args= -p $W 2>/dev/null)" = "bash /workspace/refe-plan-vdac/refe/vdac_pause_p7.sh 247636 247978 A5HB" ] || { say "after_hb: REFUSE, wrapper $W args changed"; exit 1; }
say "after_hb: sequencer3 stopped by the MM; watching A5HB wrapper $W; A5HC held for the PI (cap)"
while kill -0 $W 2>/dev/null; do sleep 30; done
say "A5HB wrapper ended: $(grep -E 'P7_RESUMED|P7_HELD|PAUSE_STOPPED_EARLY|did not finish|ZZVDAC' $OUT/pause.log | tail -3 | tr '\n' ' ' | cut -c1-400)"
say "not launching A5HC: over the PI's 16.7 L40-h cap at HB-s0's measured step cost; awaiting the PI"
