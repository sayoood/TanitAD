#!/bin/bash
# Stage the stack-sync file set off the G: mount ONE FILE AT A TIME, verifying
# each by md5 against a FRESH read, and retrying until two independent reads of
# the source agree with the staged copy.
#
# ⛔ WHY NOT `tar -T list`: MEASURED 2026-09-06. tar over the mount printed
# "Read error ... Invalid request code" on 3 files, still wrote all 44 entries,
# left ZERO zero-length files, and produced 29 of 44 with contents that did not
# match the manifest. Exit status, entry count and file size were all clean.
# The only thing that caught it was a content comparison after extraction.
#
# ⚠️ And a single md5 of the source is not enough either: the manifest and the
# tar were BOTH read through the same flaking channel, so "they disagree" does
# not say which is wrong. The loop below requires TWO SEPARATE reads of the
# source to agree with each other AND with the staged copy before a file is
# accepted.
set -uo pipefail
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
SD="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f407bc82-7969-457c-a947-6be2014fee89/scratchpad"
LIST="$SD/stack_sync_list.txt"
STAGE="$SD/stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"
ok=0; bad=0
: > "$SD/stage_report.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  mkdir -p "$STAGE/$(dirname "$f")"
  good=0
  for try in 1 2 3 4 5 6 7 8; do
    cp "$REPO/$f" "$STAGE/$f" 2>/dev/null || { sleep 2; continue; }
    a=$(md5sum "$REPO/$f" 2>/dev/null | cut -d' ' -f1)
    b=$(md5sum "$REPO/$f" 2>/dev/null | cut -d' ' -f1)   # second INDEPENDENT read
    c=$(md5sum "$STAGE/$f" 2>/dev/null | cut -d' ' -f1)
    if [ ${#a} -eq 32 ] && [ "$a" = "$b" ] && [ "$a" = "$c" ]; then
      good=1; echo "$c  $f" >> "$SD/stage_report.txt"; break
    fi
    sleep 2
  done
  if [ "$good" -eq 1 ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "FAILED  $f" >> "$SD/stage_report.txt"; fi
done < "$LIST"
echo "staged_ok=$ok  failed=$bad  of $(grep -c . "$LIST")"
[ "$bad" -eq 0 ] || exit 1
