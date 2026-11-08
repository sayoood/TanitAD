#!/bin/bash
# check_bases.sh [<tip sha>]  -- read-only. For every modified repo file in v3/blobs.tsv, is the tip's blob still the recorded base?
# Prints one line per file and ZZBASES_OK / ZZBASES_MOVED. Run right after `git fetch` and a pinned `rev-parse FETCH_HEAD`.
G=C:/Users/Admin/tanitad-push/.git
B=origin/agent/arch-inf-20260803
TIP=${1:-$(git --git-dir=$G rev-parse $B)}
REMOTE=$(git --git-dir=$G ls-remote origin agent/arch-inf-20260803 | cut -f1)
echo "tip(local ref)=$TIP  remote=$REMOTE"
[ "$TIP" = "$REMOTE" ] || echo "NOTE: local ref != remote; fetch first and pass the pinned sha"
bad=0
while IFS=$'\t' read -r p b f s; do
  case "$p" in "("*) continue;; esac
  [ "$b" = NEW ] && { cur=$(git --git-dir=$G cat-file -e "$TIP:$p" 2>/dev/null && echo EXISTS || echo absent); [ "$cur" = absent ] && echo "ok-new  $p" || { echo "NEW-BUT-EXISTS $p"; bad=1; }; continue; }
  cur=$(git --git-dir=$G rev-parse "$TIP:$p" 2>/dev/null)
  if [ "$cur" = "$b" ]; then echo "same    ${p##*/}"; else echo "MOVED   ${p##*/} base=$b tip=$cur"; bad=1; fi
done < "$(dirname "$0")/../blobs.tsv"
[ $bad = 0 ] && echo ZZBASES_OK || echo ZZBASES_MOVED
