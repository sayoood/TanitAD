#!/bin/bash
# Persistent retry for mktree_commit under sibling contention. RC=2 means the
# compare-and-swap lost a race (HEAD moved) -- that is the tool working, and the
# only correct response is to rebuild on the new HEAD and try again. Any other
# non-zero RC is a real error and stops the loop.
SD="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f407bc82-7969-457c-a947-6be2014fee89/scratchpad"
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
MSG="$1"; shift
cd "$REPO" || exit 9
for i in $(seq 1 40); do
  PYTHONIOENCODING=utf-8 "$PY" stack/scripts/mktree_commit.py "$MSG" "$@" \
    > "$SD/commit_retry.log" 2>&1
  RC=$?
  echo "[$(date -u +%FT%TZ)] attempt $i rc=$RC" >> "$SD/commit_retry.status"
  if [ "$RC" -eq 0 ]; then echo "COMMIT_OK after $i attempts" >> "$SD/commit_retry.status"; exit 0; fi
  if [ "$RC" -ne 2 ]; then
    echo "COMMIT_HARD_FAIL rc=$RC" >> "$SD/commit_retry.status"
    tail -5 "$SD/commit_retry.log" >> "$SD/commit_retry.status"
    exit "$RC"
  fi
  sleep 7
done
echo "COMMIT_GAVE_UP after 40 attempts" >> "$SD/commit_retry.status"
exit 3
