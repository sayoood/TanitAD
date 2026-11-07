#!/usr/bin/env bash
# Master Mind 2026-11-07: wait for the deciding C0C-RE pull line in renc/c0cre/logs/c0cre_chain.out, then run c0cre_read.sh once.
# The card's section 6 readings were fixed in PREREG_T2_RENC.md's log before this pull existed. MK_FAIL, a chain failure line, or 5 h
# without the pull: note it and run nothing. Notes go to logs/c0cre_read_waiter.notes only (never into the watched chain log).
# The exit code of the read is written to c0cre_read_waiter.done (one line: rc=<n>), which a dev-box background task waits on.
set -u
S=/e/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/stage5/C0C-RE
W=/e/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/renc/c0cre/logs/c0cre_chain.out
N=$S/logs/c0cre_read_waiter.notes
D=$S/c0cre_read_waiter.done
mkdir -p $S/logs
note() { echo "[$(date +%T)] $*" >> "$N"; }
[ ! -e "$D" ] || { note "done marker exists, STOP"; exit 3; }
END=$(( $(date +%s) + 5*3600 ))
note "waiting for the C0C-RE MK_DONE pull line in $W"
while :; do
  [ $(date +%s) -lt $END ] || { note "STOP: no C0C-RE pull in 5 h"; echo "rc=9" > "$D"; exit 9; }
  pl=$(tr -d '\r' < "$W" | grep -E '^ZZ STAGE5 PULLED MK_(DONE|FAIL) C0C-RE -> ' | tail -1)
  [ -n "$pl" ] && break
  fl=$(tr -d '\r' < "$W" | grep -E '^ZZ (C0CRE-RUN-FAILED|C0CRE-REFUSED|C0CRE-BUSY|STAGE5-)' | tail -1)
  [ -z "$fl" ] || { note "STOP: chain failure line: $fl"; echo "rc=8" > "$D"; exit 8; }
  sleep 60
done
note "pull line: $pl"
case "$pl" in
  *MK_FAIL*) note "STOP: stage 5 marker MK_FAIL; the read is not run"; echo "rc=7" > "$D"; exit 7 ;;
esac
bash $S/c0cre_read.sh < /dev/null > $S/c0cre_read.run.log 2>&1
rc=$?
note "c0cre_read.sh rc=$rc (log c0cre_read.run.log)"
echo "rc=$rc" > "$D"
