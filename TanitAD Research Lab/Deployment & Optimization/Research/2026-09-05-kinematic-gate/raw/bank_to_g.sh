#!/bin/sh
# Copy this package's artifacts to the G: repo with per-file SHA256 verification and a
# retry loop. ⛔ The mount drops CONTENT while serving METADATA, so a copy that "succeeds"
# can land 0 bytes -- verify by hash, never by exit code, and treat a short/absent read as
# INCONCLUSIVE rather than as a failed copy.
set -u
SRC="/c/Users/Admin/kingate"
DST="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-kinematic-gate"
mkdir -p "$DST/raw" 2>/dev/null
rc=0
for rel in "$@"; do
  s="$SRC/$rel"
  [ -s "$s" ] || { echo "SKIP-EMPTY-SRC $rel"; rc=1; continue; }
  a=$(sha256sum "$s" 2>/dev/null | cut -d' ' -f1)
  if [ ${#a} -ne 64 ]; then echo "INCONCLUSIVE-SRC-HASH $rel"; rc=1; continue; fi
  ok=0
  i=0
  while [ $i -lt 30 ]; do
    i=$((i+1))
    mkdir -p "$(dirname "$DST/$rel")" 2>/dev/null
    cp "$s" "$DST/$rel" 2>/dev/null
    b=$(sha256sum "$DST/$rel" 2>/dev/null | cut -d' ' -f1)
    if [ ${#b} -eq 64 ] && [ "$a" = "$b" ]; then
      echo "VERIFIED $rel ${a}"
      ok=1; break
    fi
    sleep 3
  done
  [ $ok -eq 1 ] || { echo "FAILED $rel (30 attempts)"; rc=1; }
done
exit $rc
