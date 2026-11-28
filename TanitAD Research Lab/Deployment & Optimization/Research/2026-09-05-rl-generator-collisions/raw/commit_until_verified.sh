#!/bin/sh
# Commit and KEEP RETRYING UNTIL THE CONTENT IS VERIFIED IN HEAD.
#
# ⛔ WHY THIS EXISTS. `mktree_commit.py` printed
#     "git mktree kept failing: [3221225478]"      (0xC0000006 STATUS_IN_PAGE_ERROR)
# and then EXITED 0. The task notification duly reported "exit code 0" and the commit had not
# happened. That is the documented "exit codes are not evidence" trap, and it is exactly the
# defect I fixed in `sep` an hour earlier wearing different clothes: a success signal that is
# not a check on the thing it claims.
#
# So: verify by BLOB COMPARISON, per path, with 40-character length guards on BOTH operands --
# because on this mount `git rev-parse` and `git hash-object` can BOTH return the empty string
# in an outage window, and `[ "$a" = "$b" ]` is then TRUE. Report INCONCLUSIVE, never MATCH.
set -u
G="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
MSG="$1"; shift
MAX="${MAX:-40}"

verify() {   # 0 = every path verified in HEAD
  for p in "$@"; do
    a=$(git -C "$G" rev-parse "HEAD:$p" 2>/dev/null)
    b=$(git -C "$G" hash-object "$G/$p" 2>/dev/null)
    if [ ${#a} -ne 40 ] || [ ${#b} -ne 40 ]; then echo "  INCONCLUSIVE $p"; return 2; fi
    if [ "$a" != "$b" ]; then echo "  NOT-IN-HEAD  $p"; return 1; fi
  done
  return 0
}

i=0
while [ $i -lt "$MAX" ]; do
  i=$((i + 1))
  if verify "$@" >/dev/null 2>&1; then
    echo "ZZCOMMIT-VERIFIED-attempt${i}-ZZ"; verify "$@"; exit 0
  fi
  echo "ZZATTEMPT-${i}-$(date -u +%H:%M:%S)Z-ZZ"
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe "$G/stack/scripts/mktree_commit.py" \
      "$MSG" "$@" > /c/Users/Admin/tanitad-rlgen/commit_attempt.log 2>&1
  if verify "$@" >/dev/null 2>&1; then
    echo "ZZCOMMIT-VERIFIED-attempt${i}-ZZ"; verify "$@"
    git -C "$G" log --oneline -1 2>/dev/null
    exit 0
  fi
  sleep 20
done
echo "ZZCOMMIT-FAILED-after-${MAX}-attempts-ZZ"
verify "$@"
exit 1
