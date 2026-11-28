#!/bin/bash
# pull_croprule.sh -- pulls the small text results of card ADD101 from the pod (the join record, the queue logs and markers, the read folder) as one tar over base64 and compares the
# sha256 of the tar on both sides. Reads only; on the pod it writes one new tar file in the card's own root. It opens no result value.
set -u
D=/f/TanitAD-scratch/v8proof/croprule
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/croprule10k
FILES="make_joins.out joins/CROP_JOINS_RECORD.json dry_s0.out queue_croprule_s0.log queue_croprule_s1.log START_CROPRULE10K_s0.txt START_CROPRULE10K_s1.txt CROPRULE10K_DONE_s0.txt CROPRULE10K_DONE_s1.txt read"
[ -e "$D/pod_add101" ] && { echo "REFUSED: $D/pod_add101 exists"; exit 2; }
timeout 180 bash "$PS" "cd $ROOT && [ ! -e pull_add101.tar ] && tar cf pull_add101.tar $FILES && echo \"ZZSHA \$(sha256sum < pull_add101.tar | cut -c1-64)\" && echo \"ZZB64 \$(base64 -w0 pull_add101.tar)\"" 2>/dev/null | sed '/hf_/d' | tr -d '\r' > "$D/tmp_pull_add101.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add101.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add101.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add101.tar"
got=$(sha256sum < "$D/tmp_pull_add101.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add101" && tar xf "$D/tmp_pull_add101.tar" -C "$D/pod_add101" && echo "PULL-OK ${got:0:16} $(find "$D/pod_add101" -type f | wc -l) files"
