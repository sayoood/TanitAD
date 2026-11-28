#!/bin/bash
# pull_gaprule.sh -- pulls small text results of cards ADD94 and ADD95 from the pod as one tar over base64 and compares the sha256 of the tar on both sides.
# Reads only; on the pod it writes one new tar file in the card's own root. It opens no result value.
# Usage: bash pull_gaprule.sh add94     the build output and the build record of card ADD94          -> gaprule/pod_add94
#        bash pull_gaprule.sh add95     the build, the wrapper checks, the queue logs and the read    -> gaprule/pod_add95
set -u
D=/f/TanitAD-scratch/v8proof/gaprule
PS=/c/Users/Admin/v8pod/pssh.sh
L=${1:-}
case "$L" in
  add94) ROOT=/workspace/v8proof/gaprule10k; FILES="make_joins.out joins/PLANTED_JOINS_RECORD.json" ;;
  add95) ROOT=/workspace/v8proof/gaprule10kb; FILES="make_joins.out joins/PLANTED_JOINS_RECORD.json check_j.out check_j.out.rc check_j.start check_j2.out check_j2.out.rc check_j2.start check_j2/CHECK_WRAPPER_J_DONE.txt check_j2/CHECK_WRAPPER_J.hashes check_j2/refusal_missing_join.log dry_s0.out queue_gaprule_s0.log queue_gaprule_s1.log START_GAPRULE10K_s0.txt START_GAPRULE10K_s1.txt GAPRULE10K_DONE_s0.txt GAPRULE10K_DONE_s1.txt read" ;;
  add95r2) ROOT=/workspace/v8proof/gaprule10kb; FILES="read2" ;;
  add95r3) ROOT=/workspace/v8proof/gaprule10kb; FILES="read3" ;;
  *) echo "usage: pull_gaprule.sh add94|add95|add95r2"; exit 2 ;;
esac
[ -e "$D/pod_$L" ] && { echo "REFUSED: $D/pod_$L exists"; exit 2; }
timeout 180 bash "$PS" "cd $ROOT && [ ! -e pull_$L.tar ] && tar cf pull_$L.tar $FILES && echo \"ZZSHA \$(sha256sum < pull_$L.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_$L.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_$L.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_$L.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_$L.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_$L.tar"
got=$(sha256sum < "$D/tmp_pull_$L.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_$L" && tar xf "$D/tmp_pull_$L.tar" -C "$D/pod_$L" && echo "PULL-OK ${got:0:16} $(find "$D/pod_$L" -type f | wc -l) files"
