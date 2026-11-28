#!/bin/bash
# pull_errpart.sh -- pulls the small text results of card ADD98 from the pod (the join record, the queue logs and markers, the read folder) as one tar over base64 and compares the
# sha256 of the tar on both sides. Reads only; on the pod it writes one new tar file in the card's own root. It opens no result value.
set -u
D=/f/TanitAD-scratch/v8proof/errpart
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/errpart10k
FILES="make_joins.out joins/PART_JOINS_RECORD.json dry_s0.out queue_errpart_s0.log queue_errpart_s1.log START_ERRPART10K_s0.txt START_ERRPART10K_s1.txt ERRPART10K_DONE_s0.txt ERRPART10K_DONE_s1.txt read"
[ -e "$D/pod_add98" ] && { echo "REFUSED: $D/pod_add98 exists"; exit 2; }
timeout 180 bash "$PS" "cd $ROOT && [ ! -e pull_add98.tar ] && tar cf pull_add98.tar $FILES && echo \"ZZSHA \$(sha256sum < pull_add98.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_add98.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add98.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add98.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add98.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add98.tar"
got=$(sha256sum < "$D/tmp_pull_add98.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add98" && tar xf "$D/tmp_pull_add98.tar" -C "$D/pod_add98" && echo "PULL-OK ${got:0:16} $(find "$D/pod_add98" -type f | wc -l) files"
