#!/bin/bash
# retry mm_commit until it lands or the deadline passes; the G: mount drops for minutes at a time
MSG="$1"; shift; DEADLINE=$((SECONDS + 7200))
R="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
while [ $SECONDS -lt $DEADLINE ]; do
  cd "$R" 2>/dev/null || { sleep 60; continue; }
  out=$("C:/Users/Admin/venvs/tanitad/Scripts/python.exe" stack/scripts/mm_commit.py "$MSG" "$@" 2>&1)
  echo "$out" | tail -4
  if echo "$out" | grep -q "^committed\|nothing to commit"; then echo "COMMIT_LANDED"; exit 0; fi
  if echo "$out" | grep -q "REFUSING\|mismatch"; then echo "COMMIT_REFUSED"; exit 2; fi
  sleep 90
done
echo "COMMIT_DEADLINE"; exit 3
