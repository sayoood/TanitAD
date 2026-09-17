#!/bin/sh
# The area-matched RANDOM off-agent arm, re-run with the PER-WINDOW column shift.
# ⛔ The first attempt drew a NEW placement for each of the 24 sub-frames; that patch
# jumps through the D-015 stack and adds apparent motion the lead mask does not have,
# which would confound "the lead" with "temporal coherence". It was stopped 5/141
# episodes in, its partial dump deleted, and the draw moved to ONE shift per window --
# the pre-registration's own wording ("drawn per window").
set -e
set -o pipefail
H=$(dirname "$0")
O=/c/Users/Admin/d3_out
until grep -q "QUEUE2_DONE" "$O/queue2.log" 2>/dev/null; do sleep 20; done
rm -rf "$O/randmask_dump" "$O/randmask.json"
bash "$H/maskroll.sh" randmask rand 0
echo "=== QUEUE_RAND_DONE ==="
