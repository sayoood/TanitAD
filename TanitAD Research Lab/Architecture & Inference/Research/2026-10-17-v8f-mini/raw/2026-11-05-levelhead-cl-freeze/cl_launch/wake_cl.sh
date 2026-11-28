#!/usr/bin/env bash
# LEVELHEAD arm CL waker (script polling, not model turns). Usage: wake_cl.sh wprobe | chain. One short READ-ONLY ssh per minute. REFUSES without CL_PI_GO.
#   wprobe: exits on WPROBE.done (0), WPROBE.FAILED (3), WPROBE.STOPPED / GUARD_STOP.txt (4), timeout (5).
#   chain : exits on CHAIN_CL.done (0, "CHAIN DONE") / CHAIN_CL.STOPPED (4), GUARD_STOP.txt (4), a NEW RUN_CL_*.FAILED (3), or a new RUN_CL_*.done (0, "NEW DONE (n of 2)": re-arm with CL_SEEN=n), timeout (5).
# Each poll mirrors box 2's marker directory listing into a local scratch folder (one empty file per name), so every end test is a local `[ -e a ] || [ -e b ]`, never `ls a b`.
# A marker folder that does not exist yet reads as an empty listing (WARN after 15). Test hooks (dev box): CL_SSH (the whole ssh command), CL_SLEEP, CL_WAKE_N, CL_SEEN, CL_SEEN_FAILED. Production runs set none of them.
set -u
: "${CL_PI_GO:?set CL_PI_GO to the PI go time (HH:MM): this waker refuses to contact box 2 without the go of the PI}"
MODE=${1:-}; case "$MODE" in wprobe|chain) ;; *) echo "usage: wake_cl.sh wprobe|chain"; exit 64;; esac
SSH=${CL_SSH:-"ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"}
RC='D:/tanitad/runs/v8f_mini_full/cl'
seen=${CL_SEEN:-0}; seenf=${CL_SEEN_FAILED:-0}; empty=0
MIR=$(mktemp -d); trap 'rm -rf "$MIR"' EXIT
shopt -s nullglob
for i in $(seq 1 ${CL_WAKE_N:-1500}); do
  listing=$($SSH "powershell -NoProfile -Command \"Get-ChildItem $RC -Name -ErrorAction SilentlyContinue\"" 2>/dev/null | tr -d '\r')
  rm -f "$MIR"/*
  while IFS= read -r nm; do case "$nm" in ""|*[!A-Za-z0-9._-]*) ;; *) : > "$MIR/$nm" ;; esac; done <<< "$listing"
  if [ -e "$MIR/GUARD_STOP.txt" ]; then echo "GUARD STOP"; echo "$listing" | grep '^RUN_\|^CHAIN\|^WPROBE\|^GUARD'; exit 4; fi
  if [ "$MODE" = wprobe ]; then
    if [ -e "$MIR/WPROBE.done" ]; then echo "WPROBE DONE"; exit 0; fi
    if [ -e "$MIR/WPROBE.FAILED" ]; then echo "WPROBE FAILED"; exit 3; fi
    if [ -e "$MIR/WPROBE.STOPPED" ]; then echo "WPROBE STOPPED"; exit 4; fi
  else
    fl=("$MIR"/RUN_CL_*.FAILED); dn=("$MIR"/RUN_CL_*.done); nf=${#fl[@]}; nd=${#dn[@]}
    if [ -e "$MIR/CHAIN_CL.done" ] || [ -e "$MIR/CHAIN_CL.STOPPED" ]; then
      if [ -e "$MIR/CHAIN_CL.done" ]; then echo "CHAIN DONE"; ec=0; else echo "CHAIN STOPPED (safety stop)"; ec=4; fi
      echo "$listing" | grep '^RUN_\|^CHAIN\|^GUARD'; exit $ec
    fi
    if [ "$nf" -gt "$seenf" ]; then echo "RUN FAILED: $(echo "$listing" | grep '^RUN_CL_.*\.FAILED$' | tr '\n' ' ')"; exit 3; fi
    if [ "$nd" -gt "$seen" ]; then echo "NEW DONE ($nd of 2): $(echo "$listing" | grep '^RUN_CL_.*\.done$' | tr '\n' ' ')"; exit 0; fi
  fi
  if [ -z "$listing" ]; then empty=$((empty + 1)); [ "$empty" -eq 15 ] && echo "WARN 15 consecutive empty listings (box 2 unreachable, or the marker folder does not exist yet)"; else empty=0; fi
  sleep ${CL_SLEEP:-60}
done
echo "WAKER TIMEOUT"; exit 5
