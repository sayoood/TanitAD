#!/usr/bin/env bash
# RECOVER-NEXT waker (script polling, not model turns). Exits (with a line saying why) on: CHAIN_RNEXT.done / CHAIN_RNEXT.STOPPED, a GUARD_STOP.txt, a NEW RUN_*.FAILED, or a new RUN_*.done (once per run).
# Start it as a background task / Monitor. One short ssh per minute; read-only. Exit codes: 0 chain done / new done, 3 run failed, 4 safety stop, 5 timeout.
# Each poll mirrors box 2's marker directory listing into a local scratch folder (one empty file per name), so every end test is a local `[ -e a ] || [ -e b ]`, never `ls a b` (which fails when either is absent).
# Test hooks (dev box): RNEXT_SSH (the whole ssh command), RNEXT_SLEEP, RNEXT_WAKE_N, RNEXT_SEEN, RNEXT_SEEN_FAILED. Production runs set none of them.
SSH=${RNEXT_SSH:-"ssh -o BatchMode=yes -o ConnectTimeout=20 tanitad-box2"}
RC='D:/tanitad/runs/v8f_mini_full/rnext'
seen=${RNEXT_SEEN:-0}; seenf=${RNEXT_SEEN_FAILED:-0}; empty=0
MIR=$(mktemp -d); trap 'rm -rf "$MIR"' EXIT
shopt -s nullglob
for i in $(seq 1 ${RNEXT_WAKE_N:-1500}); do
  listing=$($SSH "powershell -NoProfile -Command \"Get-ChildItem $RC -Name -ErrorAction SilentlyContinue\"" 2>/dev/null | tr -d '\r')
  rm -f "$MIR"/*
  while IFS= read -r nm; do case "$nm" in ""|*[!A-Za-z0-9._-]*) ;; *) : > "$MIR/$nm" ;; esac; done <<< "$listing"
  fl=("$MIR"/RUN_*.FAILED); dn=("$MIR"/RUN_*.done); nf=${#fl[@]}; nd=${#dn[@]}
  if [ -e "$MIR/CHAIN_RNEXT.done" ] || [ -e "$MIR/CHAIN_RNEXT.STOPPED" ]; then
    if [ -e "$MIR/CHAIN_RNEXT.done" ]; then echo "CHAIN DONE"; ec=0; else echo "CHAIN STOPPED (safety stop)"; ec=4; fi
    echo "$listing" | grep '^RUN_\|^CHAIN\|^GUARD'; exit $ec
  fi
  if [ -e "$MIR/GUARD_STOP.txt" ]; then echo "GUARD STOP"; echo "$listing" | grep '^RUN_\|^CHAIN\|^GUARD'; exit 4; fi
  if [ "$nf" -gt "$seenf" ]; then echo "RUN FAILED: $(echo "$listing" | grep '^RUN_.*\.FAILED$' | tr '\n' ' ')"; exit 3; fi
  if [ "$nd" -gt "$seen" ]; then echo "NEW DONE ($nd of 6): $(echo "$listing" | grep '^RUN_.*\.done$' | tr '\n' ' ')"; exit 0; fi
  if [ -z "$listing" ]; then empty=$((empty + 1)); [ "$empty" -eq 15 ] && echo "WARN 15 consecutive empty listings (box 2 unreachable, or the marker folder does not exist yet)"; else empty=0; fi
  sleep ${RNEXT_SLEEP:-60}
done
echo "WAKER TIMEOUT"; exit 5
