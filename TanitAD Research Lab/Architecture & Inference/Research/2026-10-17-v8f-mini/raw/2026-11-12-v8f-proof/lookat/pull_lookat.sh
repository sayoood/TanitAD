#!/bin/bash
# pull_lookat.sh -- pulls the small text results of card ADD107 from the pod (markers, queue job markers and logs, the identity control, the four fit records, the prediction meta files, the read folder) as one tar over
# base64 and compares the sha256 of the tar on both sides. Never a head file (.pt) and never a prediction file (.npz): the tar excludes both. Tar members are only those that exist.
# Reads only; on the pod it writes one new tar file in the card's own root. Refuses when the local folder lookat/pod_add107 exists. It opens no result value.
set -u
D=/f/TanitAD-scratch/v8proof/lookat
PS=/c/Users/Admin/v8pod/pssh.sh
ROOT=/workspace/v8proof/lookat
FILES="START_LOOKAT.txt DONE_LOOKAT.txt STOP_LOOKAT.txt dry_queue.out queue_lookat.out queue IDENTITY_K5.json fits/LOC_s0/FIT_RECORD.json fits/LOC_s1/FIT_RECORD.json fits/LOC_s2/FIT_RECORD.json fits/LOC_s0_perm/FIT_RECORD.json \
pred/LOC_s0_meta.json pred/LOC_s1_meta.json pred/LOC_s2_meta.json pred/LOC_s0_perm_meta.json read"
[ -e "$D/pod_add107" ] && { echo "REFUSED: $D/pod_add107 exists"; exit 2; }
LT=$(date +%Y%m%dT%H%M%S)
REMOTE='cd ROOT_ && T=$(date -u +%Y%m%dT%H%M%SZ) && N=pull_add107_$T.tar && M="" && for f in FILES_; do if [ -e "$f" ]; then M="$M $f"; fi; done && [ -n "$M" ] && tar cf $N --exclude="*.pt" --exclude="*.npz" $M && echo "ZZNAME $N" && echo "ZZSHA $(sha256sum < $N | cut -c1-64)" && echo "ZZB64 $(base64 -w0 $N)"'
REMOTE=${REMOTE/ROOT_/$ROOT}
REMOTE=${REMOTE/FILES_/$FILES}
timeout 180 bash "$PS" "$REMOTE" 2>/dev/null | sed '/hf[_]/d' | tr -d '\r' > "$D/tmp_pull_add107_$LT.txt"
want=$(grep '^ZZSHA ' "$D/tmp_pull_add107_$LT.txt" | cut -d' ' -f2)
grep '^ZZB64 ' "$D/tmp_pull_add107_$LT.txt" | cut -d' ' -f2 | base64 -d > "$D/tmp_pull_add107_$LT.tar"
got=$(sha256sum < "$D/tmp_pull_add107_$LT.tar" | cut -c1-64)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "PULL-FAILED want ${want:0:16} got ${got:0:16}"; exit 1; }
mkdir "$D/pod_add107" && tar xf "$D/tmp_pull_add107_$LT.tar" -C "$D/pod_add107" && echo "PULL-OK ${got:0:16} $(grep '^ZZNAME ' "$D/tmp_pull_add107_$LT.txt" | cut -d' ' -f2) $(find "$D/pod_add107" -type f | wc -l) files"
