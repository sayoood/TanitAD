#!/bin/bash
# Archive = MOVE from Thor to the dev box, verified by content (PI 2026-09-14: "cleanup the thor and archive non used disk content").
# Per directory from archive_list.tsv (largest first as listed): 1) md5 manifest of every regular file + symlink manifest, written
# on Thor; 2) tar stream Thor -> D:/thor_archive/2026-09-14/<same path>; 3) md5sum -c of EVERY file on the dev box; 4) only when
# all files verify AND the file count matches: remove the directory on Thor. Anything short of that leaves Thor untouched and is
# logged. Resumable: directories already VERIFIED+REMOVED in the log are skipped.
LIST="$1"
DEST=/d/thor_archive/2026-09-14
LOGD="<scratchpad>/cleanup"
LOG="$LOGD/archive_move.log"
mkdir -p "$DEST/_manifests"
while IFS=$'\t' read -r P BYTES WHY; do
  [ -n "$P" ] || continue
  grep -q "^REMOVED	$P	" "$LOG" 2>/dev/null && continue
  if [ "${ARCHIVE_PHASE:-copy}" = "remove" ]; then                    # phase 2: only what phase 1 verified, re-checked locally
    L=$(grep "^VERIFIED	$P	" "$LOG" | tail -1)
    [ -n "$L" ] || { echo "SKIP-NOT-VERIFIED	$P" >> "$LOG"; continue; }
    NFV=$(echo "$L" | cut -f4 | tr -dc '0-9'); PARENT=$(dirname "$P"); NAME=$(basename "$P"); REL=${PARENT#/home/nvidia}
    NLOC=$(cd "$DEST$REL" 2>/dev/null && find "$NAME" -type f | wc -l)
    [ -n "$NLOC" ] && [ "$NLOC" -ge "$NFV" ] || { echo "SKIP-LOCAL-COUNT	$P	$NLOC<$NFV" >> "$LOG"; continue; }
    ssh -o ConnectTimeout=20 -n tanitad-thor-wifi "rm -rf '$P' && [ ! -e '$P' ] && echo GONE" 2>>"$LOG" | grep -q GONE && printf 'REMOVED\t%s\t%s\n' "$P" "$BYTES" >> "$LOG" || echo "FAIL-REMOVE	$P" >> "$LOG"
    continue
  fi
  grep -q "^VERIFIED	$P	" "$LOG" 2>/dev/null && continue
  case "$P" in /home/nvidia/*) ;; *) echo "SKIP-BADPATH	$P" >> "$LOG"; continue;; esac
  FLAT=$(echo "${P#/home/nvidia/}" | tr '/' '__')
  PARENT=$(dirname "$P"); NAME=$(basename "$P"); REL=${PARENT#/home/nvidia}
  mkdir -p "$DEST$REL"
  T0=$(date +%s)
  # 1) manifests on Thor
  ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -n tanitad-thor-wifi "cd '$PARENT' && [ -d '$NAME' ] && find '$NAME' -type f -print0 | nice -n 10 xargs -0 md5sum > /home/nvidia/archive_manifest_$FLAT.md5 && find '$NAME' -type l -printf '%p\t%l\n' > /home/nvidia/archive_manifest_$FLAT.links && wc -l < /home/nvidia/archive_manifest_$FLAT.md5" > "$LOGD/_nfiles.tmp" 2>>"$LOG"
  NF=$(tr -dc '0-9' < "$LOGD/_nfiles.tmp")
  [ -n "$NF" ] || { echo "FAIL-MANIFEST	$P" >> "$LOG"; continue; }
  scp -q "tanitad-thor-wifi:/home/nvidia/archive_manifest_$FLAT.md5" "tanitad-thor-wifi:/home/nvidia/archive_manifest_$FLAT.links" "$DEST/_manifests/" 2>>"$LOG" || { echo "FAIL-MANIFEST-COPY	$P" >> "$LOG"; continue; }
  # 2) stream
  ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -n tanitad-thor-wifi "cd '$PARENT' && tar cf - --hard-dereference '$NAME'" 2>>"$LOG" | tar xf - -C "$DEST$REL" 2>>"$LOG"
  T1=$(date +%s)
  # 3) verify every regular file
  VOUT=$(cd "$DEST$REL" && md5sum -c --quiet "$DEST/_manifests/archive_manifest_$FLAT.md5" 2>&1 | grep -v "^$" | head -5)
  NLOCAL=$(cd "$DEST$REL" && find "$NAME" -type f | wc -l)
  T2=$(date +%s)
  MBPS=$(( BYTES / 1048576 / ( T1 - T0 > 0 ? T1 - T0 : 1 ) ))
  if [ -z "$VOUT" ] && [ "$NLOCAL" -ge "$NF" ]; then
    printf 'VERIFIED\t%s\t%s bytes\t%s files\t%s MB/s\t%s s verify\n' "$P" "$BYTES" "$NF" "$MBPS" "$((T2-T1))" >> "$LOG"
  else
    printf 'FAIL-VERIFY\t%s\tmanifest %s files, local %s\t%s\n' "$P" "$NF" "$NLOCAL" "$(echo $VOUT | head -c 300)" >> "$LOG"
  fi
done < "$LIST"
echo "ZZARCHIVE-END $(date -Is)" >> "$LOG"
