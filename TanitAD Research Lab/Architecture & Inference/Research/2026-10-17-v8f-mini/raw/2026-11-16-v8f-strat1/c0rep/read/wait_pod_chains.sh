#!/bin/bash
# MM overnight waiter: wakes when the C0REP chain or the DSCALE block-3 chain writes an end line (whichever comes first), polling every 10 min (script polling, no model turns).
# Baselines are taken at start, so an end line already present does not fire. Last line: ZZ WAITPOD END ZZ <which> | ZZ WAITPOD TIMEOUT ZZ
source C:/Users/Admin/lorapod/podenv.sh
F='post-quantum|store now|upgraded'
q() { pssh "$1" 2>/dev/null | grep -vE "$F" | tr -d '\r'; }
c0() { q "grep -cE '^ZZ STRAT1 C0REP (DONE|FAIL) ZZ' /root/strat1/logs/c0rep.log 2>/dev/null || echo 0" | tail -n 1; }
b3() { q "grep -cE '^ZZ DSCALE10K BLOCK3 (DONE|FAIL) ZZ' /root/dscale10k/block3.log 2>/dev/null || echo 0" | tail -n 1; }
C0=$(c0); B0=$(b3); C0=${C0:-0}; B0=${B0:-0}
echo "baseline c0rep=$C0 block3=$B0 $(date -u +%FT%TZ)"
for i in $(seq 1 144); do
  sleep 600
  c=$(c0); b=$(b3)
  if [ "${c:-0}" -gt "$C0" ]; then q "tail -n 4 /root/strat1/logs/c0rep.log"; echo "ZZ WAITPOD END ZZ c0rep"; exit 0; fi
  if [ "${b:-0}" -gt "$B0" ]; then q "tail -n 4 /root/dscale10k/block3.log"; echo "ZZ WAITPOD END ZZ block3"; exit 0; fi
done
echo "ZZ WAITPOD TIMEOUT ZZ (24 h)"
