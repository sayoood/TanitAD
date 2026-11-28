#!/bin/bash
# mm_wait_ki.sh -- MM 2026-11-23. Read-only polls of the LoRA pod every 60 s, from rh_ki_chain.log.
#  Wait for "ZZ ROLLHIST-KI " (the chain's last line). Exit 0 = DONE; 1 = FAIL; 5 = the lock has been held for 0.9 h with no end
#  (the card's return line to the PI); 3 = cap 4 h.
q() { timeout 90 bash -c 'source C:/Users/Admin/lorapod/podenv.sh; pssh "$0"' "$1" < /dev/null 2>/dev/null; }
CL=/root/rollhist/rh_ki_chain.log
for i in $(seq 1 240); do
  T=$(q "tail -n 8 $CL; echo; date -u +%s; X=\$(grep 'GPU queue lock held to the end' $CL | tail -n 1 | cut -d' ' -f1); if [ -n \"\$X\" ]; then date -u -d \"\$X\" +%s; else echo 0; fi")
  if printf '%s\n' "$T" | grep -q "^ZZ ROLLHIST-KI DONE ZZ"; then echo "KI DONE"; printf '%s\n' "$T"; exit 0; fi
  if printf '%s\n' "$T" | grep -q "^ZZ ROLLHIST-KI FAIL"; then echo "KI FAIL"; printf '%s\n' "$T"; exit 1; fi
  NOW=$(printf '%s\n' "$T" | tail -n 2 | head -n 1); LK=$(printf '%s\n' "$T" | tail -n 1)
  if [ -n "$LK" ] && [ "$LK" -gt 0 ] 2>/dev/null && [ $((NOW - LK)) -gt 3240 ]; then echo "KI LOCK ABOVE 0.9 h ($((NOW - LK)) s), no end yet"; printf '%s\n' "$T"; exit 5; fi
  sleep 60
done
echo "END WAIT TIMEOUT"; exit 3
